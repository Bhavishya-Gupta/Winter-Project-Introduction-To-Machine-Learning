import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- QUESTION 1 ----------
arr = np.random.randint(low=1, high=101, size=(4,4,4))
print(arr)

mean = np.mean(arr)
dev = np.std(arr)

for mat in arr:
    for row in mat:
        for el in row: 
            z_score = (el-mean)/dev
            print(z_score) 



# ---------- QUESTION 2 ----------
df1 = pd.read_csv('NON IITK\Assignments\Assignment 1\Iris.csv')
df2 = df1.loc[df1['PetalLengthCm'] > 3.0]
print(df2)

# ---------- QUESTION 3 ----------
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,4,9,16,25,36,49,64,81,100]

plt.plot(x,y, label="Squares of Numbers")
plt.title("Squares of Numbers")
plt.xlabel("Number")
plt.ylabel("Number^2")

plt.legend()
plt.grid()
plt.show()

# ---------- QUESTION 4 ----------
data = {
    "A": np.random.randint(1, 101, 100),
    "B": np.random.randint(1,1000, 100),
    "C": [np.nan if i%10==0 else np.random.randint(1,11) for i in range(100)]
}
dataframe = pd.DataFrame(data)
dataframe.fillna(dataframe.mean(numeric_only=True), inplace=True)
print(dataframe)

dataframe.sort_values(by=["A", "B"], ascending=[False, True], inplace=True)
print(dataframe)


# ---------- QUESTION 5 ----------
mat1 = np.random.randint(1, 100, (3,3))
mat2 = np.random.randint(1, 100, (3,3))
dotProd = mat1.dot(mat2)
determinant = np.linalg.det(dotProd)
print(determinant)

# ---------- QUESTION 6 ----------
cricket = {'Team'   : ['India', 'India', 'Australia', 'Australia', 'South Africa', 'South Africa', 'South Africa', 'South Africa', 
                       'New Zealand', 'New Zealand', 'New Zealand', 'India'],
           'Rank'   : [2,3,1,2,3,4,1,1,2,4,1,2],
           'Year'   : [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
           'Points' : [876,801,891,815,776,784,834,824,758,691,883,782]}

dfCricket = pd.DataFrame(cricket)
groupedDf = dfCricket.groupby('Team')
groupSum = groupedDf['Points'].sum()
print(groupSum)



# ---------- QUESTION 7 ----------
histData = np.random.randn(100000)
plt.hist(histData, bins=100, color='green', edgecolor='black', alpha=0.75)

plt.grid(True, linestyle='-', alpha=0.5)
plt.title('Histogram')

plt.show()

# ---------- QUESTION 8 ----------
array = np.random.randint(1, 1001, (5,5))
subArray = array[3]

print(subArray)

# ---------- QUESTION 9 ----------
dataset1 = {
    "A": np.random.randint(0, 101, 10),
    "B": np.random.randint(200, 211, 10),
    "C": [1,2,3,4,5,6,7,8,9,10]
}
dataset2 = {
    "E": np.random.randint(101, 201, 10),
    "F": np.random.randint(200, 211, 10),
    "C": [1,2,3,4,5,6,7,8,9,10]
}

dataframe1 = pd.DataFrame(dataset1)
dataframe2 = pd.DataFrame(dataset2)
dataframeFinal = pd.merge(dataframe1, dataframe2, on="C", how="inner")
print(dataframeFinal)

# ---------- QUESTION 10 ----------
df = pd.read_csv('NON IITK\Assignments\Assignment 1\Iris.csv')
sepalLength = df['SepalLengthCm']
sepalWidth = df['SepalWidthCm']

plt.scatter(sepalLength, sepalWidth, c='red', alpha=0.5, s=35 )
plt.title("Sepal Length and Width Scatter Plot")
plt.show()