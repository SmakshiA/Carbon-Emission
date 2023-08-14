import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io
import base64

from flask import Flask, request,render_template,redirect,session,url_for

# Load the dataset
dataset = pd.read_csv("CO2emmission.csv")

# Split the dataset into input (X) and output (y) variables
X = dataset.iloc[:, [3,4,9]].values
y = dataset.iloc[:, -1].values

# handling the missing data and replace missing values with nan from numpy and replace with mean of all the other values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])


# Create a correlation matrix
corr_matrix = dataset.corr()

# Create heatmap using seaborn
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')

# Show the plot
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Linear Regression model and fit it to the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

#calculating mean absolute percentage error
mape=(mean_absolute_percentage_error(y_test, y_pred))
print("Mean absolute Error : ",mape*100)
#  y_test, pred = np.array(y_test), np.array(pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test))
print(mape)

def trees():
  time = 10.01
  #time=float(input("Enter the time-period of driving the vehicle:"))

  trees_required=int(263.32/(12.5*time))
  return trees_required
# print("Classification report")
# print(classification_report(y_test,y_pred))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method=='POST':
        input1 = request.form['input1']
        input2 = request.form['input2']
        input3 = request.form['input3']

        input1 = float(input1)
        input2 = float(input2)
        input3 = float(input3)

        y_pred = regressor.predict([[input1,input2,input3]])
        result = round(y_pred[0],2)
        return render_template('index.html', result=result)
    return render_template('index.html')

@app.route('/analysis',methods=['POST'])
def analysis():
    lst = []
    for i in range(len(dataset)):
        lst.append(i)

    data = {'Fuel Consumption City (L/100 km)':[],
            'CO2 Emissions(g/km)':[]
            }
    for i in range(len(dataset)):
        ln = dataset._get_value(lst[i],'Fuel Consumption City (L/100 km)')
        ac = dataset._get_value(lst[i],'CO2 Emissions(g/km)')
        data['Fuel Consumption City (L/100 km)'].append(ln)
        data['CO2 Emissions(g/km)'].append(ac)

    d = pd.DataFrame(data)

    kmeans = KMeans(n_clusters=3).fit(d)
    labels = kmeans.labels_
    counts = np.bincount(labels)

    fig, ax = plt.subplots()
    ax.scatter(dataset['Fuel Consumption City (L/100 km)'], dataset['CO2 Emissions(g/km)'], c=kmeans.labels_)
    centers = np.array(kmeans.cluster_centers_)
    plt.xlabel(" Fuel Consumption(L/100 km) ")
    plt.ylabel(" CO2 emission (g/km) ")
    plt.title(" Carbon Emission ")
    ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=100, color='maroon',alpha=0.8)
    plot_data = io.BytesIO()
    fig.savefig(plot_data, format='png')
    plot_data.seek(0)
    plot_url = base64.b64encode(plot_data.getvalue()).decode()
    # Render the HTML template with the plot
    result = trees()
    return render_template('graphs.html', plot_url=plot_url,result=result)

if __name__ == '__main__':
    app.run(debug=True)