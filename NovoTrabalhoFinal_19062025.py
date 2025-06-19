import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

#target
target = 'therapy_history'

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
sns.countplot(x=target, data=dados)
plt.title("Distribuição da variável alvo (therapy_history)")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("grafico_classe.png")
plt.show()


#Gráfico da distribuição do atributo  classe
bins = [0,18,30,45,60,100]
labels = ['<18', '18-30', '31-45', '46-60', '60+']
dados['faixa_etaria'] = pd.cut(dados['age'], bins=bins, labels=labels, include_lowest=True)

#Remove valores nulos da variável therapy_history
idade_terapia = dados.dropna(subset=['therapy_history'])

plt.figure(figsize=(10, 6))
sns.countplot(data=idade_terapia, x='faixa_etaria', hue='therapy_history')
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
for col in x.columns:
    if x[col].isnull().any():
        if x[col].dtype in ['int64', 'float64']:
            valorPreencherBranco = x[col].median()
            x[col].fillna(valorPreencherBranco, inplace=True)
        else:
            valorPreencherBranco = x[col].mode()[0]
            x[col].fillna(valorPreencherBranco, inplace=True)
    else:
        print(f"Coluna '{col}' não tem dados faltando. Ótimo!")
        
#avaliando se ficou algum dado em branco. Deve ficar igual a zero
print(f"\nTotal de dados faltando após preencher: {x.isnull().sum().sum()}")