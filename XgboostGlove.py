import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Passo 1: Carregar o arquivo GloVe em um dicionário
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Passo 2: Criar vetores médios para cada texto
def vectorize_with_glove(texts, embeddings, embedding_dim=300):
    """
    Cria vetores médios para cada texto usando embeddings do GloVe.
    :param texts: Lista de textos
    :param embeddings: Dicionário de embeddings do GloVe
    :param embedding_dim: Dimensão dos embeddings do GloVe
    :return: Matriz numpy com os vetores médios
    """
    stopwords_list = stopwords.words('portuguese')
    text_vectors = []

    for text in texts:
        words = text.split()
        word_vectors = [
            embeddings[word]
            for word in words
            if word in embeddings and word not in stopwords_list
        ]
        if word_vectors:  # Se houver palavras válidas
            text_vectors.append(np.mean(word_vectors, axis=0))
        else:  # Se nenhuma palavra for encontrada no GloVe
            text_vectors.append(np.zeros(embedding_dim))
    
    return np.array(text_vectors)

# Passo 3: Treinar o modelo XGBoost
def train_xgb_model(X_train, y_train):
    xgb_model = XGBClassifier(eta=0.3, gamma=1, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Passo 4: Avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Passo 5: Grid Search com GloVe
def perform_grid_search(X_train, y_train):
    param_grid = {
        'eta': [0, 0.3, 1],
        'gamma': [0.1, 1, 10]
    }
    xgb_model = XGBClassifier(eval_metric='logloss')
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Melhores parâmetros:", grid_search.best_params_)
    print("Melhor accuracy:", grid_search.best_score_)
    return grid_search.best_estimator_

# Passo 6: Função principal
def main():
    # Carregar os embeddings do GloVe
    glove_file_path = 'glove.6B.300d.txt'  # Caminho para o arquivo GloVe
    embeddings = load_glove_embeddings(glove_file_path)

    # Carregar os dados
    train_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')

    X_train = train_data['text']
    y_train = train_data['label']
    X_test = test_data['text']
    y_test = test_data['label']

    # Criar vetores médios usando GloVe
    embedding_dim = 300  # Certifique-se de usar a dimensão correspondente ao GloVe
    X_train_glove = vectorize_with_glove(X_train, embeddings, embedding_dim)
    X_test_glove = vectorize_with_glove(X_test, embeddings, embedding_dim)

    # Treinar e avaliar o modelo
    xgb_model = train_xgb_model(X_train_glove, y_train)
    evaluate_model(xgb_model, X_test_glove, y_test)

    # Grid Search
    best_model = perform_grid_search(X_train_glove, y_train)
    evaluate_model(best_model, X_test_glove, y_test)

if __name__ == "__main__":
    main()
