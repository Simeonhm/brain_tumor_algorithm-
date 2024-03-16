from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn

randomstate = 42


def onehot(df, feat):
    # encode aanmaken
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # tijdelijk dataframe maken met de transformed data van df[feat]
    temp_df = pd.DataFrame(encoder.fit_transform(df[[feat]]), columns=encoder.get_feature_names_out())
    # transformed data toevoegen aan originele dataframe
    df = pd.concat([df.reset_index(drop=True), temp_df], axis=1)
    # originele feature verwijderen uit dataframe
    df.drop(columns=[feat], axis=1, inplace=True)
    # dataframe teruggeven
    return df

'''data inladen in aanmaken pandas dataframe'''
filepath = r'C:\Users\Joram\OneDrive - HvA\Jaar2\Blok2\Programmeren\NKR_IKNL_breast_syntheticdata.csv'
rawdata = pd.read_csv(filepath, sep=';')
prepdata = rawdata.drop(columns=['topo', 'pm', 'mari', 'mari_uitslag', 'key_nkr', 'key_eid'])
prepdata = prepdata.dropna()

'''Variabelen omzetten naar numerieke waardes'''
# topo_sublok
sublok_categories = ['C500', 'C501', 'C502', 'C503', 'C504', 'C505', 'C506', 'C507', 'C508', 'C509']
# ct, pt
ctpt_categories = ['0', 'IS', '1M', '1A', '1B', '1C', '1', '2', '3', '4A', '4B', '4C', '4D']
# cn
cn_categories = ['0', '1', '2A', '2B', '3A', '3B', '3C']
# pn
pn_categories = ['0', '0I', '0S', '0IS', '1M', '1MS', '1A', '1AS', '1B', '1BS', '1C', '1CS', '2A', '2AS', '2B', '3A',
                 '3B', '3C']
# stadium, cstadium, pstadium
stadium_categories = ['M', '0', '1A', '1B', '1C', '2A', '2B', '2C', '3A', '3B', '3C', '4']

'''Data voorbereiden voor training'''
encoder = OrdinalEncoder(categories=[sublok_categories], handle_unknown='use_encoded_value', unknown_value=-1)
prepdata[['topo_sublok']] = encoder.fit_transform(prepdata[['topo_sublok']])
'''ct & pt'''
encoder = OrdinalEncoder(categories=[ctpt_categories], handle_unknown='use_encoded_value', unknown_value=-1)
prepdata[['ct']] = encoder.fit_transform(prepdata[['ct']])
prepdata[['pt']] = encoder.fit_transform(prepdata[['pt']])
'''cn'''
encoder = OrdinalEncoder(categories=[cn_categories], handle_unknown='use_encoded_value', unknown_value=-1)
prepdata[['cn']] = encoder.fit_transform(prepdata[['cn']])
'''pn'''
encoder = OrdinalEncoder(categories=[pn_categories], handle_unknown='use_encoded_value', unknown_value=-1)
prepdata[['pn']] = encoder.fit_transform(prepdata[['pn']])
'''stadia'''
encoder = OrdinalEncoder(categories=[stadium_categories], handle_unknown='use_encoded_value', unknown_value=-1)
prepdata[['stadium']] = encoder.fit_transform(prepdata[['stadium']])
prepdata[['cstadium']] = encoder.fit_transform(prepdata[['cstadium']])
prepdata[['pstadium']] = encoder.fit_transform(prepdata[['pstadium']])

prepdata = onehot(prepdata, 'uitgebr_chir_code')
prepdata = onehot(prepdata, 'morf')
prepdata = onehot(prepdata, 'tumsoort')

# man/vrouw verdeling omzetten naar binair
prepdata.loc[prepdata['gesl'] == 2, 'gesl'] = 0

# kolom toevoegen voor stadium4, omzetten naar binair
prepdata['stadium4'] = prepdata['stadium']
prepdata.loc[prepdata['stadium4'] != 11, 'stadium4'] = 0
prepdata.loc[prepdata['stadium4'] == 11, 'stadium4'] = 1

'''Scheiding aanmaken tussen input en output'''
# X en Y aanmaken
X = prepdata.copy()
X = X.drop(columns=['stadium', 'stadium4'])
y = prepdata['stadium4']

''' X en Y onderverdelen in train, test en val in een 60/20/20 verdeling '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomstate, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=randomstate, stratify=y_train)

'''model instellen, in dit geval RandomForestClassifier met de volgende parameters
de criteria waarop de nodes gevormd worden is entropy
de minimale samples voor een leaf moet 4 zijn'''
model = RandomForestClassifier(criterion='entropy', random_state=randomstate, min_samples_leaf=4)
model.fit(X_train, y_train)

# belangrijkheid van de features opvragen
importance = model.feature_importances_
column_names = list(X_train.columns.values)
importance_df = pd.DataFrame(importance, columns={'importance'}, index=column_names)

'''hiervan vervolgens de belangrijkste 12 selecteren om vervolgens voor het model te gebruiken
geslacht wordt hier automatisch niet meegenomen, deels om Bias te voorkomen'''
importance_df = importance_df.sort_values(by='importance', ascending=False)
top_imp = importance_df.iloc[0:12].index.values

'''Nadat de top 12 belangrijkste waardes uit zijn gekozen moet het model opnieuw getraind worden
Dit gebeurt nu met enkele hyperparameters die eerder bepaald zijn'''
X = prepdata[top_imp].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=randomstate)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=randomstate,
                                                  stratify=y_train)
model = RandomForestClassifier(criterion='gini', max_depth=2, min_samples_leaf=1, n_estimators=100)
'''Evalueren door middel van cross-validatie'''
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
print('score per fold:', scores)
print('mean score:', scores.mean())
model.fit(X_train, y_train)

'''berekenen van confusion matrix, prestatiewaardes en toepassen threshold
de threshold is handmatig bepaald en wordt hardcoded gebruikt in deze code'''
from sklearn.metrics import confusion_matrix

y_true = np.array(y_test)
# y_pred_class = model.predict(X_val)
y_pred_prob = model.predict_proba(X_test)
y_pred_class = binarize([y_pred_prob[:, 1]], threshold=0.1)[0]

'''Berekenen en printen van prestatiewaarden'''
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = metrics.accuracy_score(y_test, y_pred_class)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1 = (precision * recall) / (precision + recall) * 2

print('Accuracy: ', accuracy, '\nRecall: ', recall, '\nPrecision: ', precision, '\nF1-score: ', F1)
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print("tn: ", confusion[0, 0], "\tfp: ", confusion[0, 1], "\nfn: ", confusion[1, 0], "\t\ttp: ", confusion[1, 1])

'''Berekenen roc_curve en AUC'''
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for RandomForest classifier')
plt.xlabel = ('False Positive Rate (1 - Specificity)')
plt.ylabel = ('True Positive Rate (Recall, a.k.a. Sensitivity)')
plt.grid(True)
plt.show()

auc_model = metrics.roc_auc_score(y_test, y_pred_prob[:, 1])
print('AUC of current model: ', auc_model)
