import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    logger.warning("Biblioteca 'transformers' ou 'torch' não encontrada. Modelos BERT não serão carregados.")


# --- Classes e Funções de Análise ---

class AdvancedNLPAnalyzer:
    """Analisador avançado de PLN com múltiplos modelos BERT especializados"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.models = {}
        self.embeddings_cache = {}

    def setup_specialized_models(self):
        """
        Configura modelos BERT especializados.
        ATENÇÃO: Para otimizar o uso de memória no Streamlit Cloud,
        apenas o modelo de sentimento geral está ativo por agora.
        Considere o "lazy loading" ou modelos menores se precisar de mais.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers não disponível. Usando apenas NLTK.")
            return

        model_configs = [
            {
                'name': 'general_sentiment',
                'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'description': 'Análise de sentimento geral'
            },
            # ATENÇÃO: Os modelos abaixo estão COMENTADOS para evitar problemas de memória.
            # Descomente apenas se tiver certeza que há recursos ou implemente lazy loading.
            # {
            #     'name': 'emotion_detection',
            #     'model': 'j-hartmann/emotion-english-distilroberta-base',
            #     'description': 'Deteção de emoções específicas'
            # },
            # {
            #     'name': 'stress_detection',
            #     'model': 'martin-ha/toxic-comment-model',
            #     'description': 'Deteção de stress e toxicidade'
            # },
            # {
            #     'name': 'portuguese_bert_general',
            #     'model': 'neuralmind/bert-base-portuguese-cased',
            #     'description': 'Modelo BERT geral para português (base para outras tarefas)'
            # }
        ]

        logger.info("A carregar modelos NLP especializados...")

        for config in model_configs:
            try:
                logger.info(f"Carregando {config['name']}...")
                
                # Ajuste para warnings de 'return_all_scores' em modelos de classificação
                pipeline_params = {
                    "model": config['model'],
                    "device": 0 if torch.cuda.is_available() else -1
                }

                if config['name'] == 'portuguese_bert_general':
                    # Para modelos base como BERT português, carregamos para extração de features
                    pipeline_obj = pipeline(
                        "feature-extraction",
                        tokenizer=AutoTokenizer.from_pretrained(config['model']),
                        **pipeline_params
                    )
                else:
                    # Para classificadores de texto, use top_k=None para obter todas as pontuações
                    pipeline_obj = pipeline(
                        "text-classification",
                        top_k=None, # Substitui return_all_scores=True (deprecado)
                        **pipeline_params
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
        if not text or pd.isna(text) or not str(text).strip(): # Adicionado .strip() para textos vazios
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
                    text_truncated = str(text)[:500] # Limite BERT
                    
                    if model_name == 'portuguese_bert_general':
                        embeddings = model_info['pipeline'](text_truncated)
                        if embeddings and embeddings[0]:
                            results[f'{model_name}_embedding_dim'] = len(embeddings[0][0])
                        continue # Pula o processamento de classificação para este modelo

                    predictions = model_info['pipeline'](text_truncated)

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

class BurnoutPredictor:
    """Modelo híbrido para prever burnout."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        # Inicializa um modelo vazio, será treinado ou carregado
        self.model = None 

    def train_hybrid_model(self, df: pd.DataFrame) -> Dict:
        """
        Treina um modelo híbrido de classificação de burnout.
        Assume que 'df' contém características e a coluna 'burnout_status'.
        """
        if df.empty:
            logger.warning("DataFrame vazio fornecido para treino do modelo.")
            return {"status": "falha", "mensagem": "DataFrame vazio."}

        # Verifique se 'burnout_status' existe e codifique-o
        if 'burnout_status' not in df.columns:
            logger.error("Coluna 'burnout_status' não encontrada no DataFrame para treino.")
            return {"status": "falha", "mensagem": "Coluna 'burnout_status' não encontrada."}

        y_encoded = self.label_encoder.fit_transform(df['burnout_status'])
        
        # Features (excluindo 'burnout_status' e colunas de texto cru que não são features numéricas)
        # Ajuste esta lista de colunas conforme as suas features reais
        features_to_use = [col for col in df.columns if col not in ['burnout_status', 'question_1_text', 'question_2_text']]
        
        # Certifique-se de que todas as features são numéricas e trate NaNs
        X = df[features_to_use].apply(pd.to_numeric, errors='coerce').fillna(0) # Substitua NaN por 0 ou outro método

        if X.empty or len(X) < 2: # Mínimo para split sem erro
            logger.warning("DataFrame de features vazio ou com menos de 2 amostras para treino.")
            return {"status": "falha", "mensagem": "Dados insuficientes para features."}

        # Verificar se há variância nos dados para train_test_split com stratify
        # A verificação robusta já está em main(), mas este é um fallback
        if len(y_encoded) < 2 or len(self.label_encoder.classes_) < 2:
            logger.warning("Dados insuficientes para estratificação ou apenas uma classe de burnout presente.")
            return {"status": "falha", "mensagem": "Dados insuficientes ou apenas uma classe de burnout."}
            
        # Verificar contagem mínima por classe para stratify
        class_counts = pd.Series(y_encoded).value_counts()
        if any(count < 2 for count in class_counts):
            logger.error(f"Não há amostras suficientes para todas as classes para estratificação. Contagens: {class_counts.to_dict()}")
            return {"status": "falha", "mensagem": "Amostras insuficientes por classe para estratificação."}

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        # Treino de modelos individuais
        clf1 = RandomForestClassifier(random_state=42)
        clf2 = GradientBoostingClassifier(random_state=42)
        clf3 = LogisticRegression(random_state=42, solver='liblinear') # solver para evitar warnings

        eclf1 = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3)], voting='hard')
        eclf1 = eclf1.fit(X_train, y_train)
        y_pred = eclf1.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist() # Para serializar

        self.model = eclf1 # Armazena o modelo treinado

        return {
            "status": "sucesso",
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": cm,
            "classes": self.label_encoder.classes_.tolist()
        }

    def predict_burnout(self, user_data: Dict) -> str:
        """Faz a predição de burnout para um novo utilizador."""
        if self.model is None:
            logger.warning("Modelo preditivo não treinado/carregado. Não é possível fazer a previsão.")
            return "Modelo não disponível para previsão."
        
        # Crie um DataFrame a partir dos dados do utilizador
        # Esta parte precisa corresponder às features usadas no treino
        # Exemplo: user_data pode ter 'idade', 'horas_trabalho', 'qualidade_sono', etc.
        # Assegure-se que as chaves de user_data correspondem às features_to_use
        
        # Filtra user_data para incluir apenas features numéricas relevantes para o modelo
        # Ajuste 'features_for_prediction' para corresponder às suas features reais
        features_for_prediction = [
            'idade', 'horas_trabalho', 'horas_lazer', 'qualidade_sono', 
            'nivel_apoio_social', 'sentimento_geral_vader', 'stress_nivel',
            # Adicione aqui outras features numéricas que usa no treino
            'general_sentiment_neg', 'general_sentiment_neu', 'general_sentiment_pos',
        ]
        
        # Crie um DataFrame com uma única linha para a previsão
        input_df = pd.DataFrame([user_data])
        
        # Assegure-se que as colunas correspondem às features usadas no treino
        # Se alguma feature estiver faltando, adicione-a com um valor padrão (e.g., 0)
        # e garanta que sejam numéricas
        X_predict = input_df[features_for_prediction].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if X_predict.empty:
            logger.error("DataFrame de entrada para previsão vazio após pré-processamento.")
            return "Erro na previsão: dados de entrada inválidos."

        prediction_encoded = self.model.predict(X_predict)
        prediction_label = self.label_encoder.inverse_transform(prediction_encoded)
        
        return prediction_label[0]

# --- Funções de Ajuda e Widgets ---

@st.cache_data
def load_nlp_models():
    """Carrega o analisador NLP com os modelos."""
    analyzer = AdvancedNLPAnalyzer()
    analyzer.setup_specialized_models()
    return analyzer

@st.cache_resource # Pode ser @st.cache_data dependendo da mutabilidade
def load_predictor():
    """Carrega o preditor de burnout."""
    predictor = BurnoutPredictor()
    # No futuro, aqui você carregaria um modelo PRÉ-TREINADO
    # Ex: predictor.model = joblib.load('caminho/para/o/seu/modelo.pkl')
    return predictor

def generate_recommendations(prediction: str) -> List[str]:
    """Gera recomendações baseadas na predição de burnout."""
    recommendations = []
    if prediction == "Alto": # Supondo que "Alto" seja a label para burnout
        recommendations.append("Procure ajuda profissional. Um terapeuta ou psicólogo pode oferecer estratégias e apoio personalizados.")
        recommendations.append("Priorize o autocuidado: reserve tempo para atividades relaxantes e hobbies.")
        recommendations.append("Revise sua carga de trabalho: delegue tarefas se possível e defina limites claros.")
        recommendations.append("Conecte-se com amigos e familiares: o apoio social é fundamental.")
        recommendations.append("Considere técnicas de relaxamento como meditação ou mindfulness.")
    elif prediction == "Moderado":
        recommendations.append("Observe os sinais de stress e tome medidas proativas para gerir a pressão.")
        recommendations.append("Melhore a qualidade do seu sono e a sua alimentação.")
        recommendations.append("Faça pausas regulares durante o trabalho e considere exercícios físicos leves.")
        recommendations.append("Invista em atividades que lhe tragam prazer e relaxamento.")
    else: # Baixo ou Neutro
        recommendations.append("Mantenha seus hábitos saudáveis e continue a monitorizar seus níveis de stress.")
        recommendations.append("Continue a investir em atividades de lazer e autocuidado.")
        recommendations.append("Promova um ambiente de trabalho e vida equilibrado.")
    return recommendations

# --- Layout da Aplicação Streamlit ---

def main():
    st.set_page_config(
        page_title="Dashboard de Análise de Burnout",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inicialização do estado da sessão
    if 'user_form_data' not in st.session_state:
        st.session_state['user_form_data'] = {}
    if 'submitted_questionnaire' not in st.session_state:
        st.session_state['submitted_questionnaire'] = False
    if 'analysis_output' not in st.session_state:
        st.session_state['analysis_output'] = None
    if 'prediction_output' not in st.session_state:
        st.session_state['prediction_output'] = None
    if 'show_results_section' not in st.session_state:
        st.session_state['show_results_section'] = False
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = []
    if 'model_training_results' not in st.session_state:
        st.session_state['model_training_results'] = None

    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para", ["Início", "Questionário", "Resultados", "Recomendações"])

    analyzer_instance = load_nlp_models()
    predictor = load_predictor()

    # --- Páginas da Aplicação ---
    if page == "Início":
        st.title("Dashboard de Análise de Burnout")
        st.markdown("""
        Bem-vindo ao Dashboard de Análise de Burnout.
        Esta aplicação ajuda a analisar o risco de burnout através de um questionário e processamento de linguagem natural (PLN).
        """)
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR3y52g-4q0P97xY1-f4cK8_c_g0J8_m_b0Zg&s", use_column_width=True)
        st.markdown("---")
        st.header("Processamento de Respostas Abertas (NLP)")
        
        # Demonstração de análise de texto (opcional na página inicial)
        if analyzer_instance.models:
            demo_text = st.text_area("Insira um texto para demonstração da análise de sentimento:", 
                                     "Estou muito cansado e stressado com o trabalho ultimamente.")
            if st.button("Analisar Texto Demo"):
                if demo_text.strip():
                    demo_analysis = analyzer_instance.analyze_text_comprehensive(demo_text)
                    st.write("Resultados da Análise de Sentimento (VADER):")
                    st.json(demo_analysis)
                else:
                    st.warning("Por favor, insira um texto para analisar.")
        else:
            st.info("Modelos NLP não carregados. A análise de texto não está disponível.")

        st.header("Modelo Preditivo de Burnout")
        
        # --- TREINO DO MODELO (DEBUG/DESENVOLVIMENTO) ---
        # ATENÇÃO: Em produção, o modelo deveria ser PRÉ-TREINADO e apenas CARREGADO aqui.
        # Este bloco é para testes, assumindo que df_final seria populado com dados
        # de múltiplas submissões de questionário ou carregado de um arquivo.
        
        # Exemplo de como df_final poderia ser criado (para teste)
        # Em uma aplicação real, df_final viria de um DB ou upload de arquivo.
        if 'all_user_data' not in st.session_state:
            st.session_state['all_user_data'] = []
        
        # Se houver dados de questionários submetidos
        if st.session_state['all_user_data']:
            df_final = pd.DataFrame(st.session_state['all_user_data'])
            # Assegure-se que as colunas numéricas são tratadas (e.g., para o modelo)
            # Exemplo de tratamento para colunas numéricas do questionário
            numeric_cols = [
                'idade', 'horas_trabalho', 'horas_lazer', 'qualidade_sono',
                'nivel_apoio_social', 'vader_compound', 'general_sentiment_neg', 
                'general_sentiment_neu', 'general_sentiment_pos', 'stress_nivel' # Assumindo que stress_nivel é do NLP
            ]
            for col in numeric_cols:
                if col in df_final.columns:
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0) # Trate NaNs
            
            # Crie uma coluna de 'burnout_status' para exemplo, baseada em algum critério
            # Isso é CRÍTICO para o treino do modelo. No seu caso, pode vir do questionário ou de outro lugar.
            # EXTREMAMENTE IMPORTANTE: A sua coluna 'burnout_status' ou 'target_variable' deve ter valores.
            # Por exemplo, se a sua coluna é "nivel_burnout" no questionário:
            # df_final['burnout_status'] = df_final['nivel_burnout'] # Use a coluna real do seu questionário
            # Para fins de teste e demonstração se não tiver, vamos criar uma dummy:
            if 'burnout_status' not in df_final.columns or df_final['burnout_status'].isnull().all():
                # Isto é uma DUMMY. Substitua pela sua lógica REAL.
                # Se não tem uma coluna real de burnout, o treino não será significativo.
                df_final['burnout_status'] = np.random.choice(['Baixo', 'Moderado', 'Alto'], size=len(df_final))
                logger.warning("Coluna 'burnout_status' não encontrada ou vazia, usando dados dummy para treino. Substitua pela sua lógica real.")

            # Verifique as colunas de 'df_final' para ter certeza que correspondem ao que o preditor espera
            # st.write("DataFrame para treino (df_final):", df_final.head())
            # st.write("Contagem de classes para 'burnout_status':", df_final['burnout_status'].value_counts())

            # ATENÇÃO: A coluna 'burnout_status_encoded' será criada dentro de train_hybrid_model
            # via LabelEncoder, mas ela precisa de 'burnout_status' como input.
            
            # Nova verificação robusta para treino
            if not df_final.empty and 'burnout_status' in df_final.columns:
                # O LabelEncoder será chamado dentro de train_hybrid_model, que cria y_encoded
                # A verificação de contagem de classes é feita ali também.
                y_target_for_check = df_final['burnout_status']
                class_counts_for_check = y_target_for_check.value_counts()
                
                min_samples_per_class_for_train = 2 # Mínimo absoluto para estratificação
                
                if (len(df_final) >= 5 and # Pelo menos 5 amostras totais
                    len(class_counts_for_check) > 1 and # Pelo menos 2 classes
                    all(count >= min_samples_per_class_for_train for count in class_counts_for_check)): # Pelo menos 2 de cada classe
                    
                    try:
                        st.write("A tentar treinar o modelo preditivo...")
                        model_training_results = predictor.train_hybrid_model(df_final)
                        st.session_state['model_training_results'] = model_training_results
                        if model_training_results["status"] == "sucesso":
                            st.success("Modelo preditivo treinado com sucesso!")
                        else:
                            st.error(f"Erro ao treinar o modelo: {model_training_results['mensagem']}")
                            st.info("Verifique se há dados suficientes e variados para o treinamento.")
                    except ValueError as e:
                        st.error(f"Erro ao treinar o modelo: Dados insuficientes ou desequilibrados para treinamento com estratificação. Detalhes: {e}")
                        st.info("Para treinar o modelo preditivo, são necessários mais dados (pelo menos 5 amostras no total e 2 amostras para cada categoria de burnout).")
                    except Exception as e:
                        st.error(f"Ocorreu um erro inesperado durante o treino do modelo: {e}")
                else:
                    st.info("Dados insuficientes para treinar o modelo preditivo. São necessários mais questionários e pelo menos 2 exemplos para cada categoria de burnout.")
                    st.session_state['model_training_results'] = None
            else:
                st.warning("Não há dados válidos de 'burnout_status' em 'df_final' para o treino do modelo.")
                st.session_state['model_training_results'] = None
        else:
            st.info("Ainda não há dados de questionários submetidos para treino do modelo.")
            st.session_state['model_training_results'] = None

    elif page == "Questionário":
        st.title("Questionário de Avaliação de Burnout")
        st.markdown("Por favor, preencha o questionário para nos ajudar a analisar seu perfil.")

        with st.form("burnout_questionnaire"):
            st.subheader("Informações Demográficas")
            idade = st.slider("Idade", 18, 70, 30)
            genero = st.radio("Género", ["Masculino", "Feminino", "Outro"])

            st.subheader("Carga de Trabalho e Equilíbrio")
            horas_trabalho = st.slider("Quantas horas por semana você dedica ao trabalho/estudo?", 10, 80, 40)
            horas_lazer = st.slider("Quantas horas por semana você dedica ao lazer e hobbies?", 0, 40, 10)
            qualidade_sono = st.slider("Qual a qualidade do seu sono? (1=Péssima, 5=Excelente)", 1, 5, 3)
            nivel_apoio_social = st.slider("Qual o seu nível de apoio social (amigos, família)? (1=Baixo, 5=Alto)", 1, 5, 3)
            
            st.subheader("Percepção Pessoal e Sentimentos")
            question_1_text = st.text_area("Descreva em algumas frases como se tem sentido ultimamente em relação ao seu trabalho/estudo e vida pessoal:", 
                                           height=150, help="Ex: 'Tenho-me sentido muito esgotado e sem energia.'")
            
            # Adicionar uma pergunta para o status de burnout, que será o target do modelo
            burnout_status = st.radio(
                "Como você classificaria seu nível de burnout nos últimos 6 meses?",
                ["Baixo", "Moderado", "Alto"]
            )

            submitted = st.form_submit_button("Submeter Questionário")

            if submitted:
                # Armazenar os dados do formulário
                st.session_state['user_form_data'] = {
                    'idade': idade,
                    'genero': genero,
                    'horas_trabalho': horas_trabalho,
                    'horas_lazer': horas_lazer,
                    'qualidade_sono': qualidade_sono,
                    'nivel_apoio_social': nivel_apoio_social,
                    'question_1_text': question_1_text,
                    'burnout_status': burnout_status # Armazena o status de burnout do questionário
                }
                st.session_state['submitted_questionnaire'] = True
                
                # Análise NLP da resposta aberta
                text_to_analyze = st.session_state['user_form_data'].get('question_1_text')
                
                # --- NOVO: VERIFICAÇÃO PARA TEXTO ANTES DE ANALISAR ---
                if text_to_analyze is not None and isinstance(text_to_analyze, str) and text_to_analyze.strip():
                    st.session_state['analysis_output'] = analyzer_instance.analyze_text_comprehensive(text_to_analyze)
                    
                    # Adicione os resultados do NLP aos dados do formulário para o modelo preditivo
                    # Certifique-se de que o preditor espera estas features
                    if st.session_state['analysis_output']:
                        # Exemplo de como adicionar resultados do NLP ao user_form_data
                        # Ajuste as chaves conforme o output real do seu analyzer
                        vader_compound = st.session_state['analysis_output'].get('vader_compound', 0.0)
                        general_sentiment_neg = st.session_state['analysis_output'].get('general_sentiment_neg', 0.0)
                        general_sentiment_neu = st.session_state['analysis_output'].get('general_sentiment_neu', 0.0)
                        general_sentiment_pos = st.session_state['analysis_output'].get('general_sentiment_pos', 0.0)
                        
                        # Supondo que você tenha um 'stress_nivel' do NLP, ajuste conforme necessário
                        stress_level = st.session_state['analysis_output'].get('stress_detection_score', 0.0) 

                        st.session_state['user_form_data'].update({
                            'vader_compound': vader_compound,
                            'general_sentiment_neg': general_sentiment_neg,
                            'general_sentiment_neu': general_sentiment_neu,
                            'general_sentiment_pos': general_sentiment_pos,
                            'stress_nivel': stress_level # Exemplo
                        })
                else:
                    st.warning("A resposta de texto está vazia. A análise de PLN não será realizada.")
                    st.session_state['analysis_output'] = {} # Garante que o output não é None
                    # Preencha features NLP com 0 ou valores padrão para o modelo preditivo
                    st.session_state['user_form_data'].update({
                        'vader_compound': 0.0,
                        'general_sentiment_neg': 0.0,
                        'general_sentiment_neu': 0.0,
                        'general_sentiment_pos': 0.0,
                        'stress_nivel': 0.0
                    })


                # Faz a predição de burnout para os dados do utilizador
                # Certifique-se de que predictor.predict_burnout espera user_form_data no formato correto
                st.session_state['prediction_output'] = predictor.predict_burnout(st.session_state['user_form_data'])
                
                # Gerar recomendações
                if st.session_state['prediction_output']:
                    st.session_state['recommendations'] = generate_recommendations(st.session_state['prediction_output'])
                
                st.session_state['show_results_section'] = True
                
                # Adicionar os dados da submissão atual ao histórico de dados
                st.session_state['all_user_data'].append(st.session_state['user_form_data'])

                st.success("Questionário submetido! Navegue para 'Resultados'.")
                st.experimental_rerun() # Para forçar a atualização dos resultados

    elif page == "Resultados":
        st.title("Resultados da Análise")
        if st.session_state['show_results_section'] and st.session_state['prediction_output']:
            st.subheader("Sua Predição de Risco de Burnout:")
            st.write(f"Com base nas suas respostas, a predição do seu nível de burnout é: **{st.session_state['prediction_output']}**")

            if st.session_state['analysis_output']:
                st.subheader("Análise de Sentimento do seu Texto:")
                st.json(st.session_state['analysis_output']) # Ou formate de forma mais amigável

            if st.session_state['model_training_results']:
                st.subheader("Resultados do Treino do Modelo (se aplicável):")
                if st.session_state['model_training_results']['status'] == "sucesso":
                    st.write(f"F1-Score: {st.session_state['model_training_results']['f1_score']:.2f}")
                    # Mais detalhes do relatório de classificação e matriz de confusão
                    st.json(st.session_state['model_training_results']['classification_report'])
                    st.write("Matriz de Confusão:", st.session_state['model_training_results']['confusion_matrix'])
                    st.write("Classes:", st.session_state['model_training_results']['classes'])
                else:
                    st.warning(st.session_state['model_training_results']['mensagem'])
            else:
                st.info("O modelo preditivo ainda não foi treinado devido a dados insuficientes ou erro. Preencha mais questionários.")

        else:
            st.info("Por favor, preencha o questionário primeiro para ver os resultados.")

    elif page == "Recomendações":
        st.title("Recomendações Personalizadas")
        if st.session_state['recommendations']:
            st.subheader(f"Recomendações baseadas na sua predição de '{st.session_state['prediction_output']}':")
            for i, rec in enumerate(st.session_state['recommendations']):
                st.write(f"{i+1}. {rec}")
        elif st.session_state['prediction_output']: # Se há predição mas não há recomendações (algo deu errado ou é nulo)
            st.info("Não foi possível gerar recomendações para este perfil.")
        else:
            st.info("Ainda não há dados de questionários submetidos para gerar recomendações. Por favor, preencha o questionário primeiro.")

if __name__ == "__main__":
    main()
    
    st.markdown("---")
    st.header("🔒 Acesso Aos Resultados Detalhados")

    ADMIN_PASSWORD = "oD5Gdq380szGh0U" # <<< COMENTÁRIO: MUDE ESTA SENHA! Escolha uma forte e única.
    st.warning("ATENÇÃO: A senha do administrador está codificada no código-fonte. Para produção, use `st.secrets` ou variáveis de ambiente para maior segurança.")

    password_input = st.text_input("Insira a senha para visualizar os resultados:", type="password")

    if password_input == ADMIN_PASSWORD:
        st.success("Senha correta! Exibindo os detalhes da análise.")
        # APRESENTE AQUI OS SEUS RESULTADOS REAIS
        # Por exemplo:
        if 'show_results_section' in st.session_state and st.session_state['show_results_section']:
            st.write("---")
            st.subheader("Resultados Detalhados (Apenas Admin):")
            st.json(st.session_state['analysis_output']) # Ou st.write(st.session_state['analysis_output'])
            st.write(f"**Predição de Burnout:** {st.session_state['prediction_output']}")
            st.write("---")
            st.subheader("Dados do Último Questionário Submetido:")
            st.json(st.session_state['user_form_data'])
            st.write("---")
            st.subheader("Histórico de Dados de Questionários (all_user_data):")
            st.dataframe(pd.DataFrame(st.session_state['all_user_data'])) # Exibe todos os dados coletados
            # Adicione gráficos, tabelas, etc. aqui
        else:
            st.info("Envie o questionário primeiro para gerar os resultados.")
    elif password_input: # Se o utilizador digitou algo mas a senha está errada
        st.error("Senha incorreta. Acesso negado.")
