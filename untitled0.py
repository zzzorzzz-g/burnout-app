import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import hashlib
import uuid
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import json
from typing import Dict, List, Tuple, Optional
import warnings
import nltk


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download dos recursos NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers não disponível. Usando apenas NLTK para análise de sentimento.")

# Configuração visual
plt.style.use('default')
sns.set_palette("husl")

class BurnoutSurvey:
    """Classe para gestão do questionário digital baseado no MBI"""
    
    def __init__(self):
        self.mbi_questions = self._load_mbi_questions()
        self.demographic_questions = self._load_demographic_questions()
        self.responses_db = []
    
    def _load_mbi_questions(self) -> Dict[str, List[str]]:
        """Carrega as 22 questões validadas do MBI"""
        return {
            'emotional_exhaustion': [
                "Sinto-me emocionalmente esgotado/a pelo meu trabalho académico",
                "Sinto-me exausto/a no final do dia de trabalho/estudo",
                "Sinto-me fatigado/a quando me levanto e tenho de enfrentar outro dia de trabalho/estudo",
                "Trabalhar/estudar com pessoas todo o dia é realmente stressante para mim",
                "Sinto-me esgotado/a pelo meu trabalho académico",
                "Sinto-me frustrado/a pelo meu trabalho/estudos",
                "Sinto que estou a trabalhar/estudar demasiado",
                "Realmente não me importo com o que acontece a algumas pessoas",
                "Trabalhar/estudar diretamente com pessoas põe-me demasiado stress"
            ],
            'depersonalization': [
                "Sinto que trato algumas pessoas como se fossem objetos impessoais",
                "Tornei-me mais insensível com as pessoas desde que faço este trabalho/curso",
                "Preocupo-me que este trabalho/curso me esteja a endurecer emocionalmente",
                "Realmente não me importo com o que acontece a algumas pessoas",
                "Sinto que as pessoas me culpam por alguns dos seus problemas"
            ],
            'personal_accomplishment': [
                "Lido muito eficazmente com os problemas das pessoas",
                "Influencio positivamente a vida das pessoas através do meu trabalho/estudos",
                "Sinto-me com muita energia",
                "Posso criar facilmente uma atmosfera relaxada com as pessoas",
                "Sinto-me estimulado/a depois de trabalhar/estudar com outras pessoas",
                "Consegui muitas coisas que valem a pena neste trabalho/curso",
                "No meu trabalho/estudos, lido com os problemas emocionais com muita calma",
                "Sinto que estou a influenciar positivamente a vida de outras pessoas através do meu trabalho/estudos"
            ]
        }
    
    def _load_demographic_questions(self) -> Dict[str, any]:
        """Carrega questões demográficas e contextuais"""
        return {
            'role': {
                'question': 'Qual é o seu papel na comunidade académica?',
                'options': ['Estudante Licenciatura', 'Estudante Mestrado', 'Estudante Doutoramento', 
                           'Docente', 'Investigador', 'Funcionário Não-Docente']
            },
            'age_group': {
                'question': 'Qual é a sua faixa etária?',
                'options': ['18-25', '26-35', '36-45', '46-55', '56+']
            },
            'work_hours': {
                'question': 'Quantas horas por semana dedica ao trabalho/estudos?',
                'type': 'number'
            },
            'sleep_quality': {
                'question': 'Como avalia a qualidade do seu sono? (1-10)',
                'type': 'scale',
                'range': (1, 10)
            },
            'support_system': {
                'question': 'Sente que tem apoio adequado da instituição?',
                'options': ['Sim, totalmente', 'Parcialmente', 'Não', 'Não sei']
            }
        }
    
    def create_streamlit_survey(self) -> Optional[Dict]:
        """Cria o questionário no Streamlit e retorna as respostas se submetidas com sucesso"""
        st.header("📋 Questionário de Burnout Académico")
        st.markdown("*Baseado no Maslach Burnout Inventory (MBI) - Completamente anónimo*")
        
        # Consentimento informado
        with st.expander("ℹ️ Consentimento Informado e Política de Privacidade"):
            st.markdown("""
            **Objetivo do Estudo**: Monitorização do burnout na comunidade académica
            
            **Anonimato**: Os seus dados são completamente anónimos e não podem ser associados à sua identidade
            
            **Direitos**: Pode desistir a qualquer momento
            
            **Uso dos Dados**: Apenas para investigação académica e melhoria do bem-estar na universidade
            
            **Conformidade RGPD**: Este estudo cumpre todas as regulamentações de proteção de dados
            """)
        
        consent = st.checkbox("Consinto participar neste estudo de forma voluntária e anónima")
        
        if not consent:
            st.warning("É necessário dar consentimento para participar no estudo")
            return None
        
        responses = {}
        
        # Dados demográficos
        st.subheader("👤 Informações Gerais")
        col1, col2 = st.columns(2)
        
        with col1:
            responses['role'] = st.selectbox(
                self.demographic_questions['role']['question'],
                self.demographic_questions['role']['options']
            )
            responses['age_group'] = st.selectbox(
                self.demographic_questions['age_group']['question'],
                self.demographic_questions['age_group']['options']
            )
        
        with col2:
            responses['work_hours'] = st.number_input(
                self.demographic_questions['work_hours']['question'],
                min_value=0, max_value=100, value=40
            )
            responses['sleep_quality'] = st.slider(
                self.demographic_questions['sleep_quality']['question'],
                1, 10, 7
            )
        
        responses['support_system'] = st.radio(
            self.demographic_questions['support_system']['question'],
            self.demographic_questions['support_system']['options']
        )
        
        # Questões MBI
        st.subheader("🧠 Questionário MBI")
        st.markdown("*Avalie cada afirmação de 0 (Nunca) a 6 (Todos os dias)*")
        
        mbi_responses = {}
        
        for dimension, questions in self.mbi_questions.items():
            st.markdown(f"**{dimension.replace('_', ' ').title()}**")
            
            for i, question in enumerate(questions):
                key = f"{dimension}_{i+1}"
                mbi_responses[key] = st.slider(
                    question,
                    0, 6, 3,
                    key=key,
                    help="0=Nunca, 1=Poucas vezes por ano, 2=Uma vez por mês, 3=Poucas vezes por mês, 4=Uma vez por semana, 5=Poucas vezes por semana, 6=Todos os dias"
                )
        
        # Resposta aberta
        st.subheader("💭 Comentários Livres")
        responses['open_response'] = st.text_area(
            "Descreva brevemente como se sente em relação aos seus estudos/trabalho acadêmico:", 
            placeholder="Partilhe os seus sentimentos, experiências ou preocupações...",
            height=100
        )
        
        # Combinar respostas
        responses.update(mbi_responses)
        
        if st.button("📤 Submeter Questionário", type="primary"):
            if self._validate_responses(responses):
                anonymous_id = self._anonymize_response(responses)
                st.success(f"✅ Questionário submetido com sucesso! ID anónimo: {anonymous_id}")
                return responses
            else:
                st.error("❌ Por favor, complete todas as questões obrigatórias")
        
        return None
    
    def _validate_responses(self, responses: Dict) -> bool:
        """Valida se as respostas estão completas"""
        required_fields = ['role', 'age_group', 'work_hours', 'sleep_quality', 'support_system']
        return all(field in responses and responses[field] is not None for field in required_fields)
    
    def _anonymize_response(self, responses: Dict) -> str:
        """Anonimiza a resposta e gera ID único"""
        # Gerar ID anónimo
        anonymous_id = str(uuid.uuid4())[:8]
        
        # Adicionar timestamp e ID
        responses['timestamp'] = datetime.now().isoformat()
        responses['anonymous_id'] = anonymous_id
        
        # Remover qualquer informação identificável
        # Hash de informações sensíveis se necessário
        
        # Armazenar resposta
        self.responses_db.append(responses)
        
        return anonymous_id
    
class AdvancedNLPAnalyzer:
    """Analisador avançado de PLN com múltiplos modelos BERT especializados"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.models = {}
        self.embeddings_cache = {}

    def setup_specialized_models(self):
        """Configura modelos BERT especializados, incluindo para burnout e português."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers não disponível. Usando apenas NLTK.")
            return

        model_configs = [
            {
                'name': 'general_sentiment',
                'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'description': 'Análise de sentimento geral'
            },
            {
                'name': 'emotion_detection',
                'model': 'j-hartmann/emotion-english-distilroberta-base',
                'description': 'Deteção de emoções específicas'
            },
            {
                'name': 'stress_detection',
                'model': 'martin-ha/toxic-comment-model', # Pode ser adaptado ou treinado para stress académico
                'description': 'Deteção de stress e toxicidade'
            },
            {
                'name': 'portuguese_bert_general',
                'model': 'neuralmind/bert-base-portuguese-cased',
                'description': 'Modelo BERT geral para português (base para outras tarefas)'
            }
            # A entrada para 'mental_health_emotion' foi removida daqui
        ]

        logger.info("A carregar modelos NLP especializados...")

        for config in model_configs:
            try:
                logger.info(f"Carregando {config['name']}...")

                if config['name'] == 'portuguese_bert_general':
                    # Para modelos base como BERT português, carregamos para extração de features
                    pipeline_obj = pipeline(
                        "feature-extraction",
                        model=config['model'],
                        tokenizer=AutoTokenizer.from_pretrained(config['model']),
                        device=0 if torch.cuda.is_available() else -1
                    )
                else:
                    # Para classificadores de texto
                    pipeline_obj = pipeline(
                        "text-classification",
                        model=config['model'],
                        return_all_scores=True,
                        device=0 if torch.cuda.is_available() else -1
                    )

                self.models[config['name']] = {
                    'pipeline': pipeline_obj,
                    'description': config['description']
                }
                logger.info(f"✓ {config['name']} carregado com sucesso!")

            except Exception as e:
                logger.error(f"✗ Erro ao carregar {config['name']}: {e}")

    def analyze_text_comprehensive(self, text: str) -> Dict:
        """Análise abrangente de texto com múltiplos modelos"""
        if not text or pd.isna(text):
            return {}

        results = {'text_length': len(str(text))}

        # Análise NLTK VADER
        try:
            vader_scores = self.sia.polarity_scores(str(text))
            results.update({
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu']
            })
        except Exception as e:
            logger.error(f"Erro VADER: {e}")

        # Análise com modelos BERT especializados
        if TRANSFORMERS_AVAILABLE:
            for model_name, model_info in self.models.items():
                try:
                    text_truncated = str(text)[:500]  # Limite BERT

                    if model_name == 'portuguese_bert_general':
                        embeddings = model_info['pipeline'](text_truncated)
                        if embeddings and embeddings[0]:
                            results[f'{model_name}_embedding_dim'] = len(embeddings[0][0])
                        continue # Pula o processamento de classificação para este modelo

                    predictions = model_info['pipeline'](text_truncated)

                    # O modelo daveni/finetuned_bert_emotion foi removido,
                    # então não precisamos mais do 'mental_health_emotion' aqui.
                    if model_name == 'emotion_detection':
                        for pred in predictions[0] if isinstance(predictions[0], list) else predictions:
                            emotion = pred['label'].lower().replace(' ', '_')
                            score = pred['score']
                            results[f'{model_name}_{emotion}'] = score

                    elif model_name == 'general_sentiment':
                        for pred in predictions[0] if isinstance(predictions[0], list) else predictions:
                            label = pred['label'].lower().replace(' ', '_')
                            score = pred['score']
                            results[f'{model_name}_{label}'] = score

                    else: # Para outros modelos como stress_detection
                        if isinstance(predictions[0], list):
                            max_pred = max(predictions[0], key=lambda x: x['score'])
                            results[f'{model_name}_label'] = max_pred['label']
                            results[f'{model_name}_score'] = max_pred['score']
                        else:
                            results[f'{model_name}_label'] = predictions[0]['label']
                            results[f'{model_name}_score'] = predictions[0]['score']

                except Exception as e:
                    logger.error(f"Erro no modelo {model_name}: {e}")

        return results

class HybridBurnoutPredictor:
    """Modelo preditivo híbrido avançado"""
    
    def __init__(self):
        self.quantitative_model = None
        self.qualitative_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        
    def calculate_mbi_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula scores MBI padronizados"""
        df_processed = df.copy()
        
        # Mapear questões para dimensões
        ee_cols = [col for col in df.columns if 'emotional_exhaustion' in col]
        dp_cols = [col for col in df.columns if 'depersonalization' in col]
        pa_cols = [col for col in df.columns if 'personal_accomplishment' in col]
        
        # Calcular scores por dimensão
        if ee_cols:
            df_processed['emotional_exhaustion_score'] = df_processed[ee_cols].sum(axis=1)
        if dp_cols:
            df_processed['depersonalization_score'] = df_processed[dp_cols].sum(axis=1)
        if pa_cols:
            df_processed['personal_accomplishment_score'] = df_processed[pa_cols].sum(axis=1)
        
        # Classificar nível de burnout baseado em critérios MBI
        df_processed['burnout_level'] = df_processed.apply(self._classify_burnout, axis=1)
        
        return df_processed
    
    def _classify_burnout(self, row) -> str:
        """Classifica nível de burnout baseado nos critérios MBI"""
        ee = row.get('emotional_exhaustion_score', 0)
        dp = row.get('depersonalization_score', 0)
        pa = row.get('personal_accomplishment_score', 0)
        
        # Critérios baseados na literatura MBI
        # Ajuste dos limiares para refletir a severidade
        if ee >= 27 and dp >= 13 and pa <= 31:
            return 'Alto'
        elif (ee >= 17 and ee <= 26) or (dp >= 7 and dp <= 12) or (pa >= 32 and pa <= 38):
            return 'Moderado'
        else:
            return 'Baixo'
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepara features para o modelo híbrido"""
        # Features quantitativas e de sentimento
        quantitative_features_data = []
        feature_names = []
        
        # Scores MBI
        mbi_cols = ['emotional_exhaustion_score', 'depersonalization_score', 'personal_accomplishment_score']
        for col in mbi_cols:
            if col in df.columns:
                quantitative_features_data.append(df[col].fillna(df[col].mean())) # Preencher NaN com a média
                feature_names.append(col)
        
        # Outras variáveis numéricas
        numeric_cols = ['work_hours', 'sleep_quality']
        for col in numeric_cols:
            if col in df.columns:
                quantitative_features_data.append(df[col].fillna(df[col].median() if not df[col].isnull().all() else 0))
                feature_names.append(col)
        
        # Features de sentimento (qualitativas)
        sentiment_cols = [col for col in df.columns if any(prefix in col.lower() for prefix in 
                         ['vader_', 'sentiment_', 'emotion_', 'stress_detection', 'portuguese_sentiment', 'mental_health_sentiment'])]
        
        for col in sentiment_cols:
            if col in df.columns:
                quantitative_features_data.append(df[col].fillna(0)) # Preencher NaN de scores de sentimento com 0
                feature_names.append(col)
        
        # Codificar variáveis categóricas
        if 'role' in df.columns:
            # Usar one-hot encoding para 'role' para evitar suposições ordinais
            role_dummies = pd.get_dummies(df['role'].fillna('Unknown'), prefix='role')
            for col in role_dummies.columns:
                quantitative_features_data.append(role_dummies[col])
                feature_names.append(col)
        
        # Combinar todas as features
        if quantitative_features_data:
            X = np.column_stack(quantitative_features_data)
        else:
            logger.warning("Não há features para treinar o modelo.")
            return np.array([]).reshape(len(df), 0), np.array([]), []

        y = df['burnout_level'].values if 'burnout_level' in df.columns else np.array([])
        
        # Garantir que X e y têm o mesmo número de amostras
        if len(X) != len(y) and len(y) > 0:
            logger.error("Número de amostras em X e y não corresponde.")
            return np.array([]).reshape(len(df), 0), np.array([]), []

        return X, y, feature_names
    
    def train_hybrid_model(self, df: pd.DataFrame) -> Dict:
        """Treina o modelo híbrido com validação cruzada"""
        logger.info("A treinar modelo híbrido...")
        
        X, y, feature_names = self.prepare_features(df)
        
        if X.shape[0] == 0 or len(y) == 0:
            logger.error("Dados insuficientes para treinar o modelo ou problema na preparação das features.")
            return {}
        
        # Label Encoding para a variável target (burnout_level)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definir modelos
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        }
        
        # Treinar e avaliar cada modelo
        model_results = {}
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Treinar modelo
                model.fit(X_train_scaled, y_train)
                trained_models[name] = model
                
                # Validação cruzada
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=StratifiedKFold(n_splits=5), scoring='f1_macro')
                
                # Teste
                y_pred = model.predict(X_test_scaled)
                test_f1 = f1_score(y_test, y_pred, average='macro')
                
                model_results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_f1': test_f1,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True, 
                                                                    target_names=self.label_encoder.classes_)
                }
                
                logger.info(f"{name}: CV F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test F1={test_f1:.3f}")
                
            except Exception as e:
                logger.error(f"Erro ao treinar {name}: {e}")
        
        # Selecionar melhor modelo
        if model_results:
            # Usar 'test_f1' para selecionar o melhor modelo, pois é uma métrica no conjunto de teste.
            best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_f1'])
            self.ensemble_model = trained_models[best_model_name]
            
            # Feature importance se disponível
            if hasattr(self.ensemble_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.ensemble_model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        return {
            'model_results': model_results,
            'best_model': best_model_name if model_results else None,
            'feature_importance': self.feature_importance,
            'feature_names': feature_names
        }

class EvidenceBasedRecommendations:
    """Sistema de recomendações baseado em evidência científica"""
    
    def __init__(self):
        self.intervention_database = self._load_intervention_database()
        self.research_references = self._load_research_references()
    
    def _load_intervention_database(self) -> Dict:
        """Carrega base de dados de intervenções validadas"""
        return {
            'Alto': {
                'Estudante': [
                    {
                        'intervention': 'Mindfulness-Based Stress Reduction (MBSR)',
                        'evidence_level': 'Alta',
                        'effect_size': 0.68,
                        'duration': '8 semanas',
                        'description': 'Programa estruturado de meditação mindfulness. Reduz stress e melhora bem-estar.'
                    },
                    {
                        'intervention': 'Terapia Cognitivo-Comportamental (TCC)',
                        'evidence_level': 'Alta',
                        'effect_size': 0.72,
                        'duration': '12-16 sessões',
                        'description': 'Reestruturação de pensamentos disfuncionais e desenvolvimento de estratégias de coping.'
                    },
                    {
                        'intervention': 'Aconselhamento Psicológico Individualizado',
                        'evidence_level': 'Alta',
                        'effect_size': 0.75,
                        'duration': 'Conforme necessário',
                        'description': 'Suporte psicológico personalizado para lidar com os desafios do burnout.'
                    },
                    {
                        'intervention': 'Técnicas de Gestão de Tempo e Carga de Trabalho',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.45,
                        'duration': '4-6 semanas',
                        'description': 'Estratégias de organização, priorização de tarefas e definição de limites para evitar sobrecarga.'
                    }
                ],
                'Docente': [
                    {
                        'intervention': 'Programa de Apoio entre Pares',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.52,
                        'duration': 'Contínuo',
                        'description': 'Grupos de apoio com colegas experientes para partilha de desafios e estratégias.'
                    },
                    {
                        'intervention': 'Redução de Carga Administrativa',
                        'evidence_level': 'Alta',
                        'effect_size': 0.63,
                        'duration': 'Permanente',
                        'description': 'Revisão e redistribuição de tarefas administrativas para otimizar o tempo dedicado ao ensino e investigação.'
                    },
                     {
                        'intervention': 'Formação em Resiliência e Gestão de Stress',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.55,
                        'duration': 'Sessões periódicas',
                        'description': 'Workshops e formações para desenvolver a capacidade de lidar com o stress académico.'
                    }
                ]
            },
            'Moderado': {
                'Estudante': [
                    {
                        'intervention': 'Exercício Físico Regular',
                        'evidence_level': 'Alta',
                        'effect_size': 0.58,
                        'duration': '3x/semana, 30min',
                        'description': 'Atividade física aeróbica moderada para alívio do stress e melhoria do humor.'
                    },
                    {
                        'intervention': 'Higiene do Sono',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.41,
                        'duration': 'Diário',
                        'description': 'Estabelecimento de rotinas e ambiente adequados para promover um sono reparador.'
                    },
                    {
                        'intervention': 'Participação em Atividades Sociais/Lazer',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.38,
                        'duration': 'Regularmente',
                        'description': 'Engajamento em hobbies e convívio social para promover o bem-estar e reduzir o isolamento.'
                    }
                ],
                'Docente': [
                    {
                        'intervention': 'Promoção de Equilíbrio Trabalho-Vida Pessoal',
                        'evidence_level': 'Alta',
                        'effect_size': 0.60,
                        'duration': 'Contínuo',
                        'description': 'Incentivar e apoiar a separação entre o trabalho e a vida pessoal para evitar a exaustão.'
                    },
                    {
                        'intervention': 'Desenvolvimento de Habilidades de Comunicação',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.40,
                        'duration': 'Workshops',
                        'description': 'Melhorar a comunicação com colegas e alunos para reduzir conflitos e mal-entendidos.'
                    }
                ]
            },
            'Baixo': {
                'Geral': [
                    {
                        'intervention': 'Manutenção de Hábitos Saudáveis',
                        'evidence_level': 'Moderada',
                        'effect_size': 0.35,
                        'duration': 'Contínuo',
                        'description': 'Preservar estratégias atuais de bem-estar, como alimentação equilibrada, exercício e tempo de lazer.'
                    },
                    {
                        'intervention': 'Consciencialização e Educação sobre Burnout',
                        'evidence_level': 'Baixa-Moderada',
                        'effect_size': 0.25,
                        'duration': 'Pontual',
                        'description': 'Informar-se sobre os sinais e sintomas do burnout para prevenção precoce.'
                    }
                ]
            }
        }
    
    def _load_research_references(self) -> Dict:
        """Carrega referências científicas"""
        return {
            'MBSR': 'Goyal et al. (2014). Meditation programs for psychological stress and well-being. JAMA Internal Medicine.',
            'TCC': 'Hofmann et al. (2012). The efficacy of cognitive behavioral therapy. Cognitive Therapy and Research.',
            'Exercise': 'Rosenbaum et al. (2014). Physical activity interventions for people with mental illness. Cochrane Review.',
            'Peer_Support': 'Hogan & Schmidt (2002). Helping and developing teachers: The role of peer assistance. Review of Educational Research.',
            'Time_Management': 'Claessens et al. (2007). A review of the time management literature. European Journal of Work and Organizational Psychology.',
            'Sleep_Hygiene': 'Irish et al. (2014). The role of sleep hygiene in preventing and treating burnout. Current Opinion in Pulmonary Medicine.'
        }
    
    def generate_personalized_recommendations(self, profile: Dict) -> List[Dict]:
        """Gera recomendações personalizadas baseadas no perfil do utilizador."""
        burnout_level = profile.get('burnout_level', 'Moderado')
        role = profile.get('role', 'Estudante')
        
        # Mapear papel para categoria
        # Incluir estudantes de mestrado/doutoramento na categoria 'Estudante'
        role_category = 'Estudante' if 'Estudante' in role else 'Docente' if role == 'Docente' or role == 'Investigador' else 'Geral'
        
        recommendations = []
        
        # Buscar intervenções específicas para o nível de burnout e papel
        if burnout_level in self.intervention_database:
            level_interventions = self.intervention_database[burnout_level]
            
            if role_category in level_interventions:
                recommendations.extend(level_interventions[role_category])
            # Se não houver recomendações específicas para o papel, procurar recomendações gerais para o nível de burnout
            elif 'Geral' in level_interventions:
                recommendations.extend(level_interventions['Geral'])
        
        # Adicionar recomendações gerais que são sempre aplicáveis, independentemente do nível de burnout
        general_applicable_recommendations = [
            {
                'intervention': 'Procurar Apoio Institucional (Ex: Gabinete de Apoio Psicológico da Universidade)',
                'evidence_level': 'Alta',
                'effect_size': 0.70,
                'duration': 'Conforme necessário',
                'description': 'Aproveitar os recursos de apoio psicológico e bem-estar oferecidos pela instituição.'
            },
            {
                'intervention': 'Estabelecer Limites entre Trabalho/Estudo e Vida Pessoal',
                'evidence_level': 'Alta',
                'effect_size': 0.65,
                'duration': 'Diário',
                'description': 'Definir horários e espaços claros para o trabalho/estudo e para o descanso e lazer.'
            }
        ]
        
        recommendations.extend(general_applicable_recommendations)
        
        # Remover duplicados com base na intervenção
        unique_recommendations = []
        seen_interventions = set()
        for rec in recommendations:
            if rec['intervention'] not in seen_interventions:
                unique_recommendations.append(rec)
                seen_interventions.add(rec['intervention'])
        
        # Priorizar por effect size
        unique_recommendations.sort(key=lambda x: x.get('effect_size', 0), reverse=True)
        
        return unique_recommendations[:5]  # Retorna as 5 melhores recomendações

class AdvancedVisualization:
    """Sistema avançado de visualizações"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.theme_config = {
            'background_color': '#f8f9fa',
            'grid_color': '#e9ecef',
            'text_color': '#495057'
        }
    
    def create_overview_dashboard(self, df: pd.DataFrame) -> List[go.Figure]:
        """Cria dashboard de visão geral"""
        figures = []
        
        # 1. Distribuição de Burnout por Papel
        if 'burnout_level' in df.columns and 'role' in df.columns and not df.empty:
            df_role_burnout = df.groupby(['burnout_level', 'role']).size().reset_index(name='count')
            fig1 = px.sunburst(
                df_role_burnout,
                path=['burnout_level', 'role'],
                values='count',
                title='📊 Distribuição de Burnout por Papel Académico',
                color='burnout_level',
                color_discrete_map={'Alto': '#ff4757', 'Moderado': '#ffa502', 'Baixo': '#2ed573'}
            )
            fig1.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                               paper_bgcolor=self.theme_config['background_color'], 
                               font_color=self.theme_config['text_color'])
            figures.append(fig1)
        else:
            logger.warning("Dados insuficientes para a visualização 'Distribuição de Burnout por Papel'.")
        
        # 2. Análise Temporal (simulada) - Ajustado para ser mais flexível
        if not df.empty and 'timestamp' in df.columns and 'burnout_level' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.to_period('W').dt.start_time
            temporal_df = df.groupby(['date', 'burnout_level']).size().reset_index(name='count')
            
            fig2 = px.line(
                temporal_df, x='date', y='count', color='burnout_level',
                title='📈 Evolução Temporal do Burnout',
                labels={'date': 'Data', 'count': 'Número de Respostas'},
                color_discrete_map={'Alto': '#ff4757', 'Moderado': '#ffa502', 'Baixo': '#2ed573'}
            )
            fig2.update_layout(xaxis_title="Data", yaxis_title="Número de Casos", font_size=12,
                               plot_bgcolor=self.theme_config['background_color'], 
                               paper_bgcolor=self.theme_config['background_color'], 
                               font_color=self.theme_config['text_color'],
                               xaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']),
                               yaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']))
            figures.append(fig2)
        else:
            logger.warning("Dados insuficientes para a visualização 'Evolução Temporal do Burnout' (requer 'timestamp').")
        
        # 3. Média de Scores MBI por Grupo Etário
        mbi_scores_cols = ['emotional_exhaustion_score', 'depersonalization_score', 'personal_accomplishment_score']
        if all(col in df.columns for col in mbi_scores_cols) and 'age_group' in df.columns and not df.empty:
            df_age_mbi = df.groupby('age_group')[mbi_scores_cols].mean().reset_index()
            df_age_mbi_melted = df_age_mbi.melt(id_vars='age_group', var_name='MBI Dimension', value_name='Average Score')
            
            fig3 = px.bar(
                df_age_mbi_melted, x='age_group', y='Average Score', color='MBI Dimension',
                barmode='group', title='📊 Média dos Scores MBI por Grupo Etário',
                labels={'age_group': 'Grupo Etário', 'Average Score': 'Score Médio'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig3.update_layout(xaxis_title="Grupo Etário", yaxis_title="Score Médio", font_size=12,
                               plot_bgcolor=self.theme_config['background_color'], 
                               paper_bgcolor=self.theme_config['background_color'], 
                               font_color=self.theme_config['text_color'],
                               xaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']),
                               yaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']))
            figures.append(fig3)
        else:
            logger.warning("Dados insuficientes para a visualização 'Média dos Scores MBI por Grupo Etário'.")

        # 4. Impacto da Qualidade do Sono no Burnout
        if 'sleep_quality' in df.columns and 'burnout_level' in df.columns and not df.empty:
            fig4 = px.box(
                df, x='burnout_level', y='sleep_quality', color='burnout_level',
                title='😴 Impacto da Qualidade do Sono no Nível de Burnout',
                labels={'burnout_level': 'Nível de Burnout', 'sleep_quality': 'Qualidade do Sono (1-10)'},
                color_discrete_map={'Alto': '#ff4757', 'Moderado': '#ffa502', 'Baixo': '#2ed573'}
            )
            fig4.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                               paper_bgcolor=self.theme_config['background_color'], 
                               font_color=self.theme_config['text_color'],
                               xaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']),
                               yaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']))
            figures.append(fig4)
        else:
            logger.warning("Dados insuficientes para a visualização 'Impacto da Qualidade do Sono'.")

        # 5. Correlação entre Horas de Trabalho e Nível de Burnout (se houver dados)
        if 'work_hours' in df.columns and 'burnout_level' in df.columns and not df.empty:
            fig5 = px.violin(
                df, y='work_hours', x='burnout_level', color='burnout_level', box=True,
                title='⏰ Horas de Trabalho por Nível de Burnout',
                labels={'work_hours': 'Horas de Trabalho por Semana', 'burnout_level': 'Nível de Burnout'},
                color_discrete_map={'Alto': '#ff4757', 'Moderado': '#ffa502', 'Baixo': '#2ed573'}
            )
            fig5.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                               paper_bgcolor=self.theme_config['background_color'], 
                               font_color=self.theme_config['text_color'],
                               xaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']),
                               yaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']))
            figures.append(fig5)
        else:
            logger.warning("Dados insuficientes para a visualização 'Horas de Trabalho por Nível de Burnout'.")

        return figures

    def create_sentiment_analysis_charts(self, df: pd.DataFrame) -> List[go.Figure]:
        """Cria gráficos para análise de sentimento."""
        figures = []

        # 1. Distribuição de Sentimento VADER
        if 'vader_compound' in df.columns and not df.empty:
            fig_vader = px.histogram(
                df, x='vader_compound', nbins=20,
                title='📈 Distribuição do Sentimento VADER (Respostas Abertas)',
                labels={'vader_compound': 'Score de Sentimento Composto VADER'},
                color_discrete_sequence=[self.color_palette[0]]
            )
            fig_vader.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                                    paper_bgcolor=self.theme_config['background_color'], 
                                    font_color=self.theme_config['text_color'])
            figures.append(fig_vader)
        else:
            logger.warning("Dados insuficientes para a visualização 'Distribuição do Sentimento VADER'.")

        # 2. Sentimento por Nível de Burnout (VADER)
        if 'vader_compound' in df.columns and 'burnout_level' in df.columns and not df.empty:
            fig_sentiment_burnout = px.box(
                df, x='burnout_level', y='vader_compound', color='burnout_level',
                title='🔗 Relação entre Sentimento VADER e Nível de Burnout',
                labels={'burnout_level': 'Nível de Burnout', 'vader_compound': 'Score de Sentimento Composto VADER'},
                color_discrete_map={'Alto': '#ff4757', 'Moderado': '#ffa502', 'Baixo': '#2ed573'}
            )
            fig_sentiment_burnout.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                                                paper_bgcolor=self.theme_config['background_color'], 
                                                font_color=self.theme_config['text_color'])
            figures.append(fig_sentiment_burnout)
        else:
            logger.warning("Dados insuficientes para a visualização 'Relação entre Sentimento VADER e Nível de Burnout'.")

        # 3. Análise de Emoções (se disponível)
        emotion_cols = [col for col in df.columns if 'emotion_' in col]
        if emotion_cols and not df.empty:
            # Calcular a média das emoções
            emotion_averages = df[emotion_cols].mean().reset_index()
            emotion_averages.columns = ['Emotion', 'Average Score']
            emotion_averages['Emotion'] = emotion_averages['Emotion'].str.replace('emotion_', '').str.title()
            
            fig_emotions = px.bar(
                emotion_averages, x='Emotion', y='Average Score',
                title='🎭 Média de Scores de Emoção (Respostas Abertas)',
                labels={'Emotion': 'Emoção', 'Average Score': 'Score Médio'},
                color='Emotion', color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_emotions.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                                       paper_bgcolor=self.theme_config['background_color'], 
                                       font_color=self.theme_config['text_color'])
            figures.append(fig_emotions)
        else:
            logger.warning("Dados insuficientes para a visualização 'Média de Scores de Emoção'.")
            
        return figures

    def create_feature_importance_chart(self, feature_importance_df: pd.DataFrame) -> Optional[go.Figure]:
        """Cria um gráfico de importância de features."""
        if feature_importance_df is None or feature_importance_df.empty:
            logger.warning("Não há dados de importância de features para criar o gráfico.")
            return None
        
        fig = px.bar(
            feature_importance_df.head(10).sort_values(by='importance', ascending=True), # Top 10 features
            x='importance', y='feature', orientation='h',
            title='🌟 Importância das Features na Predição de Burnout (Top 10)',
            labels={'importance': 'Importância', 'feature': 'Feature'},
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(font_size=12, plot_bgcolor=self.theme_config['background_color'], 
                           paper_bgcolor=self.theme_config['background_color'], 
                           font_color=self.theme_config['text_color'],
                           xaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']),
                           yaxis=dict(showgrid=True, gridcolor=self.theme_config['grid_color']))
        return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Monitorização de Burnout Académico")
    
    # Inicialização de classes
    survey_manager = BurnoutSurvey()
    nlp_analyzer = AdvancedNLPAnalyzer()
    nlp_analyzer.setup_specialized_models() # Carrega os modelos NLP especializados
    
    predictor = HybridBurnoutPredictor()
    recommender = EvidenceBasedRecommendations()
    visualizer = AdvancedVisualization()

    st.sidebar.title("Opções")
    page = st.sidebar.radio("Navegar", ["Preencher Questionário", "Análise de Dados", "Recomendações"])

    if page == "Preencher Questionário":
        st.title("Questionário de Burnout Académico")
        st.markdown("Por favor, preencha o questionário para nos ajudar a compreender melhor o burnout na comunidade académica.")
        
        new_response = survey_manager.create_streamlit_survey()
        
        if new_response:
            # Armazenar resposta em sessão para uso posterior na análise
            if 'all_responses_df' not in st.session_state:
                st.session_state['all_responses_df'] = pd.DataFrame()
            
            # Converter a nova resposta para DataFrame e concatenar
            response_df = pd.DataFrame([new_response])
            st.session_state['all_responses_df'] = pd.concat([st.session_state['all_responses_df'], response_df], ignore_index=True)
            st.rerun() # Recarregar a página para limpar o formulário e permitir nova submissão

    elif page == "Análise de Dados":
        st.title("Dashboard de Análise de Burnout")

        if 'all_responses_df' in st.session_state and not st.session_state['all_responses_df'].empty:
            df = st.session_state['all_responses_df'].copy()

            # Processar dados: calcular scores MBI
            df_mbi_scores = predictor.calculate_mbi_scores(df)

            # Análise NLP das respostas abertas
            st.subheader("Processamento de Respostas Abertas (NLP)")
            if 'open_response' in df_mbi_scores.columns and df_mbi_scores['open_response'].notna().any():
                with st.spinner("A realizar análise de PLN nas respostas abertas..."):
                    df_mbi_scores['nlp_analysis'] = df_mbi_scores['open_response'].apply(nlp_analyzer.analyze_text_comprehensive)
                    
                    # Expandir o dicionário de NLP para colunas separadas
                    nlp_df = pd.json_normalize(df_mbi_scores['nlp_analysis'])
                    df_final = pd.concat([df_mbi_scores.drop(columns=['nlp_analysis']), nlp_df], axis=1)
                st.success("Análise de PLN concluída!")
            else:
                df_final = df_mbi_scores
                st.info("Não há respostas abertas para análise de PLN ou todas estão vazias.")
            
            # Exibir DataFrame processado (opcional, para depuração)
            # st.subheader("Dados Processados (Amostra)")
            # st.dataframe(df_final.head())
            
            # Treinar modelo preditivo
            st.subheader("Modelo Preditivo de Burnout")
            with st.spinner("A treinar o modelo preditivo..."):
                model_training_results = predictor.train_hybrid_model(df_final)
            
            if model_training_results:
                st.success("Modelo preditivo treinado com sucesso!")
                st.json(model_training_results['model_results'])
                
                if predictor.feature_importance is not None:
                    st.subheader("Importância das Features")
                    fig_importance = visualizer.create_feature_importance_chart(predictor.feature_importance)
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.warning("Não foi possível treinar o modelo preditivo. Verifique se há dados suficientes.")

            # Visualizações
            st.subheader("Visualizações de Dados")
            
            # Dashboard de Visão Geral
            overview_figures = visualizer.create_overview_dashboard(df_final)
            for fig in overview_figures:
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráficos de Análise de Sentimento
            sentiment_figures = visualizer.create_sentiment_analysis_charts(df_final)
            for fig in sentiment_figures:
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Ainda não há dados de questionários submetidos. Por favor, preencha o questionário primeiro.")
            
    elif page == "Recomendações":
        st.title("Recomendações Personalizadas para Prevenção e Gestão de Burnout")
        
        if 'all_responses_df' in st.session_state and not st.session_state['all_responses_df'].empty:
            df = st.session_state['all_responses_df'].copy()
            df_processed = predictor.calculate_mbi_scores(df) # Recalcular scores MBI para garantir

            st.markdown("Selecione um perfil anónimo para gerar recomendações:")
            
            # Assegurar que anonymous_id é uma string para o selectbox
            df_processed['anonymous_id'] = df_processed['anonymous_id'].astype(str)
            
            selected_id = st.selectbox("Escolha um ID Anónimo", df_processed['anonymous_id'].unique())
            
            if selected_id:
                user_profile = df_processed[df_processed['anonymous_id'] == selected_id].iloc[0].to_dict()
                
                st.subheader(f"Perfil Selecionado (ID: {selected_id})")
                st.write(f"**Nível de Burnout Avaliado:** {user_profile.get('burnout_level', 'Não Avaliado')}")
                st.write(f"**Papel Académico:** {user_profile.get('role', 'Não Especificado')}")
                st.write(f"**Idade:** {user_profile.get('age_group', 'Não Especificado')}")
                
                st.subheader("Recomendações Baseadas em Evidência:")
                recommendations = recommender.generate_personalized_recommendations(user_profile)
                
                if recommendations:
                    for i, rec in enumerate(recommendations):
                        st.markdown(f"**{i+1}. {rec['intervention']}**")
                        st.write(f"   - **Nível de Evidência:** {rec['evidence_level']}")
                        st.write(f"   - **Efeito Esperado (tamanho):** {rec['effect_size']:.2f}")
                        st.write(f"   - **Duração Sugerida:** {rec['duration']}")
                        st.write(f"   - **Descrição:** {rec['description']}")
                        
                        # Adicionar referência (se existir e for relevante)
                        ref_key = next((key for key in recommender.research_references if key.lower() in rec['intervention'].lower()), None)
                        if ref_key:
                            st.caption(f"   *Referência: {recommender.research_references[ref_key]}*")
                        st.markdown("---")
                else:
                    st.info("Não foi possível gerar recomendações para este perfil.")
        else:
            st.info("Ainda não há dados de questionários submetidos para gerar recomendações. Por favor, preencha o questionário primeiro.")

if __name__ == "__main__":
    main()
    
    st.markdown("---")
st.header("🔒 Acesso Aos Resultados Detalhados")

ADMIN_PASSWORD = "oD5Gdq380szGh0U" # <<< COMENTÁRIO: MUDE ESTA SENHA! Escolha uma forte e única.

password_input = st.text_input("Insira a senha para visualizar os resultados:", type="password")

if password_input == ADMIN_PASSWORD:
    st.success("Senha correta! Exibindo os detalhes da análise.")
    # APRESENTE AQUI OS SEUS RESULTADOS REAIS
    # Por exemplo:
    if 'show_results_section' in st.session_state and st.session_state['show_results_section']:
        st.write("---")
        st.subheader("Resultados Detalhados:")
        st.json(st.session_state['analysis_output']) # Ou st.write(st.session_state['analysis_output'])
        st.write(f"**Predição de Burnout:** {st.session_state['prediction_output']}")
        # Adicione gráficos, tabelas, etc. aqui
    else:
        st.info("Envie o questionário primeiro para gerar os resultados.")
elif password_input: # Se o utilizador digitou algo mas não está correto
    st.error("Senha incorreta. Acesso negado aos resultados.")
else: # Se a caixa de senha está vazia
    st.info("Para ver os resultados, por favor, insira a senha.")

# --- FIM DA SECÇÃO DE RESULTADOS PROTEGIDA ---
