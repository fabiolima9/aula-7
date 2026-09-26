import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import MyMemoryTranslator # 1. Importação alterada

# Baixa o vocabulário do VADER
nltk.download('vader_lexicon', quiet=True)

# Inicializa o analisador
sia = SentimentIntensityAnalyzer()

# 2. Motor de tradução alterado para evitar o limite do Google
tradutor = MyMemoryTranslator(source='pt-BR', target='en-US')

st.title("Análise de Sentimentos 🎭")
st.write("Digite um texto em português para descobrir o sentimento.")

texto_usuario = st.text_area("Texto para análise:")

if st.button("Analisar Sentimento"):
    if texto_usuario:
        try:
            # Traduz o texto
            texto_traduzido = tradutor.translate(texto_usuario)
            
            # Calcula as pontuações
            resultado = sia.polarity_scores(texto_traduzido)
            pontuacao_compound = resultado['compound']
            
            if pontuacao_compound >= 0.05:
                sentimento = "Positivo 🟢"
            elif pontuacao_compound <= -0.05:
                sentimento = "Negativo 🔴"
            else:
                sentimento = "Neutro ⚪"
                
            st.write(f"**Resultado:** {sentimento}")
            st.write(f"**Pontuação (Compound):** {pontuacao_compound}")
        except Exception as e:
            # Previne que a aplicação "quebre" na tela do usuário caso a internet caia ou dê erro
            st.error("Ocorreu um erro na tradução. Tente novamente em alguns segundos.")
    else:
        st.warning("Por favor, digite um texto.")
