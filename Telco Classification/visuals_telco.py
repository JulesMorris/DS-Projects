# ignore warnings
import warnings
warnings.filterwarnings("ignore")

#linear algebra
import numpy as np
import pandas as pd

#helper modules
import prepare3
import acquire2
import visuals_telco

import matplotlib.ticker as mtick
import inflection

#visualization tools
import shap
import yellowbrick.classifier 
from yellowbrick.classifier import ROCAUC
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


from itertools import cycle, islice
from matplotlib import cm

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

################################################################################

df = acquire2.get_telco_data(use_cache = True)
telco_df = prepare3.prep_telco_data(df)
train, test, validate = telco_df



#create data copy on train to work on visualizations
data_viz = train.copy()
data_viz.loc[:, 'churn'] = data_viz.loc[:, 'churn'].apply(lambda x: 'churn' if x == 'Yes' else 'retain')

#create horizontal stacked plot bar
#import inflection
def stacked_bar_plot(data, x, y, orient = 'horizontal', ax = None, show = True):
  # create axis if not present
  if ax == None:
    _, ax = plt.subplots(figsize = (8,6))
  
  # create crosstab based on the input data
  crosstab = pd.crosstab(index = data[x], columns = data[y], normalize = 'index')
  crosstab = crosstab.reindex(['retain', 'churn'], axis = 1)

  # visualize stacked barplot
  if orient == 'vertical':
    # order in descending (the highest value on the left)
    crosstab = crosstab.sort_values('churn', ascending = False)
    crosstab.plot(kind = 'bar', stacked = True, ax = ax)

    # add percentage label
    for i, index in enumerate(crosstab.index):
        for (proportion, y_loc) in zip(crosstab.loc[index], crosstab.loc[index].cumsum()):
          ax.text(x = i,
                  y = (y_loc - proportion) + (proportion / 2),
                  s = f'{proportion*100:.1f}%',
                  color = 'white',
                  fontsize = 14,
                  fontweight = 'bold',
                  horizontalalignment = 'center',
                  verticalalignment = 'center')
    # remove tick labels
    ax.set_yticklabels([])
  else: # default is horizontal bar plot, even if the orient input is an arbitrary value
    # orient in ascending (the highest value on the top)
    crosstab = crosstab.sort_values('churn', ascending = True)
    crosstab.plot(kind = 'barh', stacked = True, ax = ax)

    # add percentage label
    for i, index in enumerate(crosstab.index):
        for (proportion, x_loc) in zip(crosstab.loc[index], crosstab.loc[index].cumsum()):
          ax.text(y = i,
                  x = (x_loc - proportion) + (proportion / 2),
                  s = f'{proportion*100:.1f}%',
                  color = 'white',
                  fontsize = 14,
                  fontweight = 'bold',
                  horizontalalignment = 'center',
                  verticalalignment = 'center')
    # remove tick labels
    ax.set_xticklabels([])

  x_titleize = inflection.titleize(x)
  ax.set_title(f'Customer Churn Probability by {x_titleize}')      
  ax.set_xlabel('')
  ax.set_ylabel('')
  ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), title = '', frameon = False)
  # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), title='', ncol=2, frameon=False)
  sns.despine(left = True, bottom = True)

  if show:
    plt.show()


##################################################################################################################

def viz1():
    pie_data = data_viz['churn'].value_counts(normalize = True).values * 100
    pie_label = data_viz['churn'].value_counts(normalize = True).index.to_list()

    fig, ax = plt.subplots(figsize = (8, 6))

    wedges, texts, autotexts = ax.pie(pie_data, labels = pie_label,
                                    startangle = 90,
                                    explode = [0, 0.1],
                                    autopct = '%.0f%%',
                                    textprops = {'color': 'w', 'fontsize':16, 'weight': 'bold'})

    for i, wedge in enumerate(wedges):
        texts[i].set_color(wedge.get_facecolor())
    plt.tight_layout()
    plt.show()

##################################################################################################################


def viz2():
    cat_col = list(train.columns)
    cat_col.remove('tenure')
    cat_col.remove('monthly_charges')
    cat_col.remove('total_charges')

    #remove redundancies
    cat_col.remove('gender_encoded')
    cat_col.remove('partner_encoded')
    cat_col.remove('dependents_encoded')
    cat_col.remove('phone_service_encoded')
    cat_col.remove('paperless_billing_encoded')
    cat_col.remove('senior_citizen_encoded')
    cat_col.remove('multiple_lines_yes')
    cat_col.remove('online_security_yes')
    cat_col.remove('device_protection_yes')
    cat_col.remove('tech_support_yes')
    cat_col.remove('streaming_tv_yes')
    cat_col.remove('streaming_movies_yes')
    cat_col.remove('contract_type_one_year')
    cat_col.remove('contract_type_two_year')
    cat_col.remove('internet_service_type_fiber_optic')
    cat_col.remove('internet_service_type_none')
    cat_col.remove('payment_type_credit_card')
    cat_col.remove('payment_type_electronic_check')
    cat_col.remove('payment_type_mailed_check')

    #remove target variables
    cat_col.remove('churn')
    cat_col.remove('churn_encoded')

    columns = data_viz['churn']

    attr_crosstab = pd.DataFrame()

    for col in cat_col: #putting cols instead of col removes churn (desired result)
        #create crosstab for each attribute
        index = data_viz[col]
        ct = pd.crosstab(index = index, columns = columns, normalize = 'index', colnames = [None]).reset_index()
        
        #add prefix to each category
        #format : column name (category)
        
        col_titleize = inflection.titleize(col)
        ct[col] = ct[col].apply(lambda x: f'{col_titleize} ({x})')
        
        #rename the column
        
        ct.rename(columns = {col: 'attribute'}, inplace = True)
        
        #create a single dataframe
        attr_crosstab = pd.concat([attr_crosstab, ct])
        
    attr_crosstab = attr_crosstab.sort_values('churn', ascending = False).reset_index(drop = True)
    #new part is setting variable equal to background style
    attr_crosstab = attr_crosstab.style.background_gradient()
    return attr_crosstab


    #################################################################################################

def viz3():
    features = ['contract_type', 'internet_service_type', 'paperless_billing', 'payment_type']

    _, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 8), sharey = True)
    for i, feature in enumerate (features):
        sns.barplot(feature, 'churn_encoded', data = train, ax = ax[i], hue = 'senior_citizen')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('Churn')
        ax[i].set_title(feature)
        ax[i].axhline(train.churn_encoded.mean(), ls = '--', color = 'red')

    sns.despine(left = True, bottom = True)
    plt.tight_layout()    
    plt.show()

###################################################################################################
def stats1():
    null = "seniority and contract types are independent."
    alt = "there is a relationship between seniority and contract types."
    α = 0.05

    #setup crosstab
    observed = pd.crosstab(train.senior_citizen, train.contract_type)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print(f'P-Value: {p/2:.3f}')
    print(f'P-value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
        #print('Reject the null hypothesis which states,', null)
        #print('Evidence suggests that', alt)
    else:
        print('Fail to reject the null hypothesis.')
        #print('Fail to reject the null:', null)
        #print('There is insufficient evidence to support the claim that,', alt)


###################################################################################################


def stats2():
    null = "seniority and payment type are independent."
    alt = "there is a relationship between seniority and payment type."
    α = 0.05

    #setup crosstab
    observed = pd.crosstab(train.senior_citizen, train.payment_type)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'P-value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
        #print('Reject the null hypothesis which states,', null)
        #print('Evidence suggests that', alt)
    else:
        print('Fail to reject the null hypothesis.')
        #print('Fail to reject the null:', null)
       # print('There is insufficient evidence to support the claim that,', alt)


 ###################################################################################################

def viz4():
    features = ['contract_type', 'internet_service_type', 'paperless_billing', 'payment_type']

    _, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 8), sharey = True)
    for i, feature in enumerate (features):
        sns.barplot(feature, 'churn_encoded', data = train, ax = ax[i], hue = 'partner')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('Churn')
        ax[i].set_title(feature)
        ax[i].axhline(train.churn_encoded.mean(), ls = '--', color = 'red')

    sns.despine(left = True, bottom = True)
    plt.tight_layout()
    plt.show()
          
 ###################################################################################################

def stats3():
    null = "being partnered and internet service types are independent."
    alt = "there is a relationship between partnership and internet service types."
    α = 0.05

    #setup crosstab
    observed = pd.crosstab(train.partner, train.internet_service_type)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'P-value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

 ###################################################################################################

def stats4():
    null = "being partnered and fiber optic internet service are independent."
    alt = "there is a relationship between being partnered and fiber optic internet service type."
    α = 0.05

    #setup crosstab
    observed = pd.crosstab(train.partner, train.internet_service_type_fiber_optic)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'P-value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')
   
 ###################################################################################################

def viz5():
    fig, ax = plt.subplots(figsize = (8, 6))
    stacked_bar_plot(data = data_viz, x = 'senior_citizen', y = 'churn', ax = ax)

###################################################################################################

def viz6():
    #Do seniors represent a higher proportion of MTM contracts? - Visualization
    colors = ['b','darkorange', 'g']
    contract_churn = train.groupby(['senior_citizen', 'contract_type']).size().unstack()

    ax = (contract_churn.T * 100.0 / contract_churn.T.sum()).T.plot(kind = 'bar',
                                                                    width = 0.3,
                                                                    stacked = True,
                                                                    rot = 0, 
                                                                    figsize = (16,8),
                                                                    color = colors)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc = 'best',prop = {'size':14},title = 'Contract Type')
    ax.set_ylabel('Customer Percentage',size = 14)
    ax.set_title('Senior Citizen Contract Type',size = 14)
    ax.set_xlabel('Senior Citizen')

    # Code to add the data labels on the stacked bar chart
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:.0f}%'.format(height), (p.get_x() + .40 * width, p.get_y() + .4 * height), #changed from .25 * width to .40 * width to center
                    color = 'white',
                    weight = 'bold',
                    size = 16)

    sns.despine(left = True, bottom = True)    
    plt.tight_layout()
    plt.show()


###################################################################################################

def viz7():
    fig, ax = plt.subplots(figsize = (8, 6))
    stacked_bar_plot(data = data_viz, x = 'partner', y = 'churn', ax = ax)


###################################################################################################

def viz8():
    my_colors = list(islice(cycle(['b', 'r', 'g', 'darkorange', 'c', 'm']), None, len(train)))
    sns.set(style = 'white')
    plt.figure(figsize = (16, 8))
    train.corr()['churn_encoded'].drop('churn_encoded').sort_values(ascending = False).plot(kind = 'bar', color = my_colors)
    sns.despine(left = True, bottom = True)
    plt.tight_layout()
    plt.show()
    #plt.xticks(rotation = 80) 


###################################################################################################
def modeling_vars(train, validate, test):
    categorical_columns = ['gender', 
                        'senior_citizen', 
                        'partner', 
                        'dependents', 
                        'phone_service', 
                        'multiple_lines', 
                        'online_security', 
                        'online_backup', 
                        'device_protection', 
                        'tech_support', 
                        'streaming_tv', 
                        'streaming_movies', 
                        'paperless_billing', 
                        'churn', 
                        'internet_service_type', 
                        'contract_type', 
                        'payment_type']


    categorical_columns.append('churn')
    categorical_columns.append('churn_encoded')


    X_train = train.drop(columns = categorical_columns)
    y_train = train.churn_encoded

    X_validate = validate.drop(columns = categorical_columns)
    y_validate = validate.churn_encoded

    X_test = test.drop(columns = categorical_columns)
    y_test = test.churn_encoded

    return X_train, y_train, X_validate, y_validate, X_test, y_test


#################################################################################################

def dt(X_train, y_train):


    #get modeling variables
    #X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    clf = DecisionTreeClassifier(max_depth = 5, random_state = 123)
    # model.fit(X , y)
    clf = clf.fit(X_train, y_train)
    #make predictions
    y_predictions1 = clf.predict(X_train)
    #estimate probability
    y_pred_proba1 = clf.predict_proba(X_train)

    print('Accuracy of Decision Tree Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

    print('Classification report, Train:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_train, y_predictions1, output_dict = True)
    return pd.DataFrame(report)

###################################################################################################
def dt_confusion():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    clf = DecisionTreeClassifier(max_depth = 5, random_state = 123)
    # model.fit(X , y)
    clf = clf.fit(X_train, y_train)
    #make predictions
    y_predictions1 = clf.predict(X_train)

    #pd.DataFrame(report), y_train, y_predictions1 = decision_tree()

    act_labels = ['Actually Retained','Actually Churned']
    col_labels = ['Pred. Retained', 'Pred. Churned']
    print('Confusion Matrix:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return pd.DataFrame(confusion_matrix(y_train, y_predictions1), index = act_labels, columns = col_labels)

###################################################################################################
def random_forest(X_train, y_train):

    rf = RandomForestClassifier(max_depth = 5, min_samples_leaf = 5,random_state = 123)

    # fit the model on train
    rf.fit(X_train, y_train)
    #make predictions
    y_predictions2 = rf.predict(X_train)
    #estimate probability
    y_pred_proba2 = rf.predict_proba(X_train)

    # Use the model 
    # We'll evaluate the model's performance on train and only train

    y_predictions = rf.predict(X_train)
    print('Accuracy of Random Forest Classifier on training set: {:.2f}'.format(rf.score(X_train, y_train)))
    print('Classification report, Train:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #produce the classification report on the y values and this models predicted y values
    report = classification_report(y_train, y_predictions2, output_dict = True)
    return pd.DataFrame(report)

####################################################################################################
def rf_confusion():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #create model
    rf = RandomForestClassifier(max_depth = 5, min_samples_leaf = 5,random_state = 123)

    # fit the model on train
    rf.fit(X_train, y_train)
    # make predictions
    y_predictions2 = rf.predict(X_train)

    labels = ['Actually Retained','Actually Churned']
    col_labels = ['Pred. Retained', 'Pred. Churned']
    print('Confusion Matrix:')
    print('~~~~~~~~~~~~~~~~~~~~~~')
    return pd.DataFrame(confusion_matrix(y_train, y_predictions2), index = labels, columns = col_labels)

####################################################################################################

def knn(X_train, y_train):

    #n_neighbour = 10

    #make model
    knn = KNeighborsClassifier(10)
    #fit model on train
    knn.fit(X_train, y_train)
    #make predictions on train
    y_predictions3 = knn.predict(X_train)

    #estimate class probability
    y_pred_proba3 = knn.predict_proba(X_train)

    print('Accuracy of K Nearest Neighbors Classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
    print('Classification report, Train:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_train, y_predictions3, output_dict = True)
    return pd.DataFrame(report)

####################################################################################################
def knn_confusion():
    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)
    #make model
    knn = KNeighborsClassifier(10)
    #fit on train
    knn.fit(X_train, y_train)
    #make predicitons on train
    y_predictions3 = knn.predict(X_train)

    labels = ['Actually Retained','Actually Churned']
    col_labels = ['Pred. Retained', 'Pred. Churned']
    print('Confusion Matrix:')
    print('~~~~~~~~~~~~~~~~~~~~~~')
    return pd.DataFrame(confusion_matrix(y_train, y_predictions3), index = labels, columns = col_labels)

# ####################################################################################################
def lr(X_train, y_train):

    #make model
    logit7 = LogisticRegression(random_state = 123)

    #select features
    features = ['internet_service_type_fiber_optic',
                'payment_type_electronic_check',
                'tenure']
    #fit on train
    logit7.fit(X_train[features], y_train)

    #make predictions on train
    y_predictions4 = logit7.predict(X_train[features])

    #get class probabilities for each datapoint on train
    y_pred_proba4 = logit7.predict_proba(X_train[features])

    print('Logistic Regression using fiber optic internet service, e-check, and tenure as features')
    print('Accuracy of Logistic Regression on training set: {:.2f}'.format(logit7.score(X_train[features], y_train)))
    print('Classification report, Train:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    #create report
    report = classification_report(y_train, y_predictions4, output_dict = True)
    return pd.DataFrame(report)

# ####################################################################################################
def lr_confusion():
    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

     #make model
    logit7 = LogisticRegression(random_state = 123)

    #select features
    features = ['internet_service_type_fiber_optic',
                'payment_type_electronic_check',
                'tenure']
    #fit on train
    logit7.fit(X_train[features], y_train)

    #make predictions
    y_predictions4 = logit7.predict(X_train[features])

    labels = ['Actually Retained','Actually Churned']
    col_labels = ['Pred. Retained', 'Pred. Churned']
    print('Confusion Matrix:')
    print('~~~~~~~~~~~~~~~~~~~~~~')
    return pd.DataFrame(confusion_matrix(y_train, y_predictions4), index = labels, columns = col_labels)


######################################################################################################
#                                           VALIDATION
# ####################################################################################################
def dt_v():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #create model
    clf = DecisionTreeClassifier(max_depth = 5, random_state = 123)
    # model.fit(X , y) on train
    clf = clf.fit(X_train, y_train)

    #predict on validate
    y_predictions1 = clf.predict(X_validate)

    #get class probabilities for each datapoint on validate
    y_pred_proba1 = clf.predict_proba(X_validate)
    print( 'Accuracy of Decision Tree Classifier on validation set: {:.4f}'.format(clf.score(X_validate, y_validate)))
    print('Classification report, Validate:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_validate, y_predictions1, output_dict = True)
    return pd.DataFrame(report)

# ####################################################################################################
def rf_v():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #make model
    rf = RandomForestClassifier(max_depth = 5, min_samples_leaf = 5,random_state = 123)

    # fit the model on train
    rf.fit(X_train, y_train)

    #make predictions on validate
    y_predictions2 = rf.predict(X_validate)

    #estimate probability
    y_pred_proba2 = rf.predict_proba(X_validate)
    print( 'Accuracy of Random Forest Classifier on validation set: {:.4f}'.format(rf.score(X_validate, y_validate)))
    print('Classification report, Validate:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_validate, y_predictions2, output_dict = True)
    return pd.DataFrame(report)

# ####################################################################################################
def knn_v():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #make model
    knn = KNeighborsClassifier(10)

    #fit model on train
    knn.fit(X_train, y_train)

    #make predictions on validate
    y_predictions3 = knn.predict(X_validate)

    #estimate class probability
    y_pred_proba3 = knn.predict_proba(X_validate)

    print( 'Accuracy of K Nearest Neighbors Classifier on validation set: {:.4f}'.format(knn.score(X_validate, y_validate)))
    print('Classification report, Validate:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_validate, y_predictions3, output_dict = True)
    return pd.DataFrame(report)

# #################################################################################################### 
def lr_v():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #make model
    logit7 = LogisticRegression(random_state = 123)

    #select features
    features = ['internet_service_type_fiber_optic',
                'payment_type_electronic_check',
                'tenure']
    #fit on train
    logit7.fit(X_train[features], y_train)

    #make predictions on validate
    y_predictions4 = logit7.predict(X_validate[features])
    #estimate class probability
    y_pred_proba4 = logit7.predict_proba(X_validate[features])
    print('Accuracy of Logistic Regression classifier on validation set: {:.4f}'
        .format(logit7.score(X_validate[features], y_validate)))

    print('Classification report, Validate:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_validate, y_predictions4, output_dict = True)
    return pd.DataFrame(report)

# ####################################################################################################
def viz_validate():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    my_model = RandomForestClassifier(max_depth = 5, min_samples_leaf = 5,random_state = 123).fit(X_train, y_train)
    #feature_names = [i for i in X_train.columns if X_train[i].dtype in [np.int64, np.int64] ]
    feature_names = X_train.columns

    explainer = shap.TreeExplainer(my_model)

    shap_values = explainer.shap_values(X_validate)

    return shap.summary_plot(shap_values[1], X_validate)

# ####################################################################################################
#                                           TEST    
# ####################################################################################################
def lr_test():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #make model
    logit7 = LogisticRegression(random_state = 123)

    #select features
    features = ['internet_service_type_fiber_optic',
                'payment_type_electronic_check',
                'tenure']
    #fit on train
    logit7.fit(X_train[features], y_train)

    #predict on test
    y_predictions5 = logit7.predict(X_test[features])

    #estimate class probability 
    y_pred_proba5 = logit7.predict_proba(X_test[features])

    print('Accuracy of Logistic Regression classifier on test set: {:.4f}'
        .format(logit7.score(X_test[features], y_test)))
    print('Classification report, Validate:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    report = classification_report(y_test, y_predictions5, output_dict = True)
    return pd.DataFrame(report)


# ####################################################################################################

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    #make model
    logit7 = LogisticRegression(random_state = 123)

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model)
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()

    return visualizer

# ####################################################################################################

def viz_auc():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #select features
    features = ['internet_service_type_fiber_optic',
                'payment_type_electronic_check',
                'tenure']
    #model
    logit7 = LogisticRegression(random_state = 123)

    #call function
    plot_ROC_curve(model = logit7, 
               xtrain = X_train[features], 
               ytrain = y_train, 
               xtest = X_test[features],
               ytest = y_test)



####################################################################################################
def get_predictions_csv():

    #get modeling variables
    X_train, y_train, X_validate, y_validate, X_test, y_test = modeling_vars(train, validate, test)

    #select features
    features = ['internet_service_type_fiber_optic',
                    'payment_type_electronic_check',
                    'tenure']
    #model
    logit7 = LogisticRegression(random_state = 123)

    #fit on train
    logit7.fit(X_train[features], y_train)

    #predict on test
    y_test_predictions = logit7.predict(X_test[features])

    #get class probability on test
    y_test_probability = logit7.predict_proba(X_test[features])

    #index = X_test.index
    predictions = pd.DataFrame({
        'predictions': y_test_predictions
    })

    #set X_test index (customer id to a new df)
    df4 = pd.DataFrame({'predictions': y_test_predictions}, index = X_test.index)

    predictions = df4
    predictions.tail()

    #get the classes of the model
    logit7.classes_

    #since only interested in (1), add new column with all the rows and column 1
    predictions['churn_probability'] = y_test_probability[:, 1]

    #confirm it worked
    predictions.head(1)

    predictions.to_csv('predictions.csv', index=False)


    predictions_df = pd.read_csv('predictions.csv')
    return predictions_df