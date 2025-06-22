import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

meuDataSet = 'dadosExcel_dataSet.xlsx'

try:
    lendo = pd.read_excel(meuDataSet)
    print("Carregado com sucesso")
except Exception  as e:
    print(f'Erro ao tentar carregar {e}')
    exit()

if not isinstance(lendo, pd.DataFrame):
    raise ValueError("Arquivo nao carregado")

#criando uma cópia de segurança
dados = lendo.copy()

#exibe as primeiras linhas do dataFrame
#print("Informações: ")
#print(dados.head())

#remove colunas que não precisam ser contabilizadas
colunas_removidas = ["id","name","country","city"]
dados = dados.drop(columns=[col for col in colunas_removidas if col in dados.columns])

#exibe as primeiras linhas do dataFrame
print("Informações: ")
print(dados.head())

#Coluna alvo - target
target = 'Historico_Terapia'

#verificar se o target está sendo reconhecido ainda
if target not in  dados.columns:
    raise ValueError("Target nao foi reconhecido")

#Caracterização dos dados
print("\nQuantidade de atributos:", dados.drop(columns=[target]).shape[1])
print("\nTipos de atributos:")
print(dados.dtypes)

print("\nValores únicos por coluna:")
for col in dados.columns:
    print(f"{col}: {dados[col].unique()[:5]}...")

#dados faltantes nas colunas
print("Colunas com dados faltantes")
print(dados.isnull().sum())

print("\nDistribuição do atributo classe:")
print(dados[target].value_counts())

print("\nNúmero total de instâncias:", dados.shape[0])

#dados para grafico - relação idade x therapy
# Gráfico da distribuição do atributo classe
plt.figure(figsize=(8, 4))
sns.countplot(x=target, data=dados, saturation=1.0, color='red')
plt.title("Distribuição da variável alvo (Historico_Terapia)")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("grafico_classe.png")
plt.show()

#Gráfico da distribuição do atributo  classe
bins = [0,18,30,45,60,100]
labels = ['<18', '18-30', '31-45', '46-60', '60+']
dados['faixa_etaria'] = pd.cut(dados['age'], bins=bins, labels=labels, include_lowest=True)

#Remove valores nulos da variável Historico_Terapia
idade_terapia = dados.dropna(subset=['Historico_Terapia'])

plt.figure(figsize=(10, 6))
sns.countplot(data=idade_terapia, x='faixa_etaria', hue='Historico_Terapia', color='orange')
plt.title("Distribuição de terapia por faixa etária")
plt.xlabel("Faixa Etária")
plt.ylabel("Quantidade de Pessoas")
plt.legend(title="Histórico de Terapia")
plt.tight_layout()
plt.savefig("grafico_idade_terapia.png")
plt.show()

#separando features x de target y
x = dados.drop(columns=target) #todas as colunas, menos meu alvo = X
y = dados[target] #somente meu alvo = Y

#print(' xxxx: ',x)
print(' Target: ',y)

#tratamento de dados em branco/faltantes
#for col in x.columns:
#    if x[col].isnull().any():
#        if x[col].dtype in ['int64', 'float64']:
#            valorPreencherBranco = x[col].median()
#            x[col].fillna(valorPreencherBranco, inplace=True)
#        else:
#            valorPreencherBranco = x[col].mode()[0]
#            x[col].fillna(valorPreencherBranco, inplace=True)
#    else:
#        print(f"Coluna '{col}' não tem dados faltando. Ótimo!")
        
#avaliando se ficou algum dado em branco. Deve ficar igual a zero
#print(f"\nTotal de dados faltando após preencher: {x.isnull().sum().sum()}")


# Codificar variáveis categóricas e booleanas
le = LabelEncoder()
for col in x.select_dtypes(include=['object', 'bool']).columns:
    x[col] = le.fit_transform(x[col])

# Codificar variável alvo
y_encoded = le.fit_transform(y)

if 'faixa_etaria' in x.columns:
    x = x.drop(columns=['faixa_etaria'])

#removendo os dados NaN
x = x.dropna()
y_encoded = y_encoded[x.index]


# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Dividir conjunto em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Definir os modelos
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Avaliação dos modelos
results = {}
print("\n=== RESULTADOS DOS MODELOS ===\n")

for model_name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results[model_name] = {
        "accuracy": report['accuracy'],
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1-score": report['macro avg']['f1-score'],
        "train_time_sec": round(end - start, 4)
    }

    # Impressão detalhada
    print(f"\nModelo: {model_name}")
    print(f"Acurácia: {report['accuracy']:.4f}")
    print(f"Precisão média: {report['macro avg']['precision']:.4f}")
    print(f"Recall médio: {report['macro avg']['recall']:.4f}")
    print(f"F1-score médio: {report['macro avg']['f1-score']:.4f}")
    print(f"Tempo de treinamento: {round(end - start, 4)} segundos")

    print("\nDesempenho por classe:")
    for label in report.keys():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"Classe {label}: F1={report[label]['f1-score']:.4f}, Precision={report[label]['precision']:.4f}, Recall={report[label]['recall']:.4f}")

# Exibir resultados em formato de tabela
print("\n=== TABELA DE COMPARAÇÃO FINAL ===")
print(f"{'Modelo':<22} {'Acurácia':<10} {'Precisão':<10} {'Recall':<10} {'F1-score':<10} {'Tempo (s)':<10}")
for model, metrics in results.items():
    print(f"{model:<22} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}    {metrics['recall']:.4f}     {metrics['f1-score']:.4f}      {metrics['train_time_sec']:.4f}")