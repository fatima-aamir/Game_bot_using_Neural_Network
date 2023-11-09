#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install tensorflow


# In[2]:


import tensorflow as tf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

#from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv("C:/Users/shame/Downloads/GameData.csv")


# In[4]:


data2=data


# In[5]:


data.to_csv('data.csv', sep=';', encoding='utf-8', index=False, header=True)


# In[6]:


data=data*1


# In[7]:


data


# In[8]:


dropColumns=['player2_move_id','player1_move_id','player2_is_player_in_move','player1_is_player_in_move','player2_is_crouching','player1_is_crouching','player2_is_jumping','player1_is_jumping','is_round_over','Timer','fight result','has_round_started','player1_id','player1_button_select','player1_button_start','player2_id','player2_button_select','player2_button_start']
data=data.drop(dropColumns,axis=1)


# In[9]:


data


# In[10]:


data.columns


# In[11]:


columns = data.columns
num_columns = len(columns)

columns1 = columns[:num_columns // 2]
columns2 = columns[num_columns // 2:]

Player1 = data[columns1]
Player2 = data[columns2]


# In[12]:


Player1


# In[13]:


Player2


# In[14]:


# Select the four columns you want to combine
col1_P1 = Player1['player1_button_up']
col2_P1 = Player1['player1_button_down']
col3_P1 = Player1['player1_button_right']
col4_P1 = Player1['player1_button_left']


# Concatenate the values from the four columns into a new column
new_column_P1 = col1_P1.astype(str) + col2_P1.astype(str) + col3_P1.astype(str) + col4_P1.astype(str)

# Insert the new column back into the original dataset using .loc
Player1.loc[:, 'Player1_Movement'] = new_column_P1


# In[15]:


Player1


# In[16]:


dropColumns_P1Movements=['player1_button_up','player1_button_down','player1_button_right','player1_button_left']
Player1=Player1.drop(dropColumns_P1Movements,axis=1)


# In[17]:


Player1


# In[18]:


# Select the four columns you want to combine
col1_P1Attack = Player1['player1_button_Y']
col2_P1Attack = Player1['player1_button_B']
col3_P1Attack = Player1['player1_button_X']
col4_P1Attack = Player1['player1_button_A']
col5_P1Attack = Player1['player1_button_L']
col6_P1Attack = Player1['player1_button_R']


# Concatenate the values from the four columns into a new column
new_column_P1 = col1_P1Attack.astype(str) + col2_P1Attack.astype(str) + col3_P1Attack.astype(str) + col4_P1Attack.astype(str)+ col5_P1Attack.astype(str)+ col6_P1Attack.astype(str)

# Insert the new column back into the original dataset using .loc
Player1.loc[:, 'Player1_Attack'] = new_column_P1


# In[19]:


dropColumns_P1Movements=['player1_button_Y','player1_button_B','player1_button_X','player1_button_A','player1_button_L','player1_button_R']
Player1=Player1.drop(dropColumns_P1Movements,axis=1)


# In[20]:


Player1


# In[21]:


Player1.info()


# In[22]:


Player1['Player1_Movements'] = Player1['Player1_Movement'].apply(lambda x: int(x, 2))
Player1['Player1_Attacks'] = Player1['Player1_Attack'].apply(lambda x: int(x, 2))


# In[23]:


Player1


# In[24]:


Player1.info()


# In[25]:


Player1=Player1.drop('Player1_Movement',axis='columns')
Player1=Player1.drop('Player1_Attack',axis='columns')


# In[26]:


Player1


# In[27]:


x=Player1['Player1_Attacks']
y=Player1['player1_health']

sns.scatterplot(y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()


# In[28]:


x=Player1['Player1_Movements']
y=Player1['player1_health']

sns.scatterplot(y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()


# In[29]:


Player2


# In[30]:


# Select the four columns you want to combine
col1_P2 = Player2['player2_button_up']
col2_P2 = Player2['player2_button_down']
col3_P2 = Player2['player2_button_right']
col4_P2 = Player2['player2_button_left']


# Concatenate the values from the four columns into a new column
new_column_P2 = col1_P2.astype(str) + col2_P2.astype(str) + col3_P2.astype(str) + col4_P2.astype(str)

# Insert the new column back into the original dataset using .loc
Player2.loc[:, 'Player2_Movement'] = new_column_P2


# In[31]:


dropColumns_P2Movements=['player2_button_up','player2_button_down','player2_button_right','player2_button_left']
Player2=Player2.drop(dropColumns_P2Movements,axis=1)


# In[32]:


Player2


# In[33]:


# Select the four columns you want to combine
col1_P2Attack = Player2['player2_button_Y']
col2_P2Attack = Player2['player2_button_B']
col3_P2Attack = Player2['player2_button_X']
col4_P2Attack = Player2['player2_button_A']
col5_P2Attack = Player2['player2_button_L']
col6_P2Attack = Player2['player2_button_R']


# Concatenate the values from the four columns into a new column
new_column_P2 = col1_P2Attack.astype(str) + col2_P2Attack.astype(str) + col3_P2Attack.astype(str) + col4_P2Attack.astype(str)+ col5_P2Attack.astype(str)+ col6_P2Attack.astype(str)

# Insert the new column back into the original dataset using .loc
Player2.loc[:, 'Player2_Attack'] = new_column_P2


# In[34]:


dropColumns_P2Movements=['player2_button_Y','player2_button_B','player2_button_X','player2_button_A','player2_button_L','player2_button_R']
Player2=Player2.drop(dropColumns_P2Movements,axis=1)


# In[35]:


Player2


# In[36]:


Player2['Player2_Movements'] = Player2['Player2_Movement'].apply(lambda x: int(x, 2))
Player2['Player2_Attacks'] = Player2['Player2_Attack'].apply(lambda x: int(x, 2))


# In[37]:


Player2=Player2.drop('Player2_Movement',axis='columns')
Player2=Player2.drop('Player2_Attack',axis='columns')


# In[38]:


Player2


# In[39]:


x=Player2['Player2_Attacks']
y=Player2['player2_health']

sns.boxplot(y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Box Plot')
plt.show()


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[41]:


mergedtables = pd.concat([Player1, Player2], axis=1)


# In[42]:


mergedtables


# In[43]:


mergedtables = pd.DataFrame(mergedtables)


# In[44]:


mergedtables.info()


# In[45]:


columns_to_convert=['player1_health','player1_x_coord','player1_y_coord','Player1_Movements','Player1_Attacks','player2_health','player2_x_coord','player2_y_coord','Player2_Movements','Player2_Attacks']

for column in columns_to_convert:
    mergedtables[column] = mergedtables[column].apply(lambda x: bin(x)[2:])


# In[46]:


# Print the updated dataset
mergedtables


# In[47]:


data2.loc[:, 'X_Coordinates'] = abs(data2['player1_x_coord'] - data2['player2_x_coord']) 


# In[48]:


data2.loc[:, 'Y_Coordinates'] = abs(data2['player1_y_coord'] - data2['player2_y_coord']) 


# In[49]:


data2=data2.drop(['has_round_started','is_round_over','player1_is_jumping','player1_is_crouching','player2_health','player1_id','player1_move_id','player2_move_id','player2_id','player1_is_player_in_move','player2_is_player_in_move','fight result','player1_button_start','player1_button_select', 'player2_button_start','player2_button_select','player1_button_right','player1_button_left','player1_button_up','player1_button_down','player1_button_L','player1_button_R','player1_button_X','player1_button_Y','player1_button_A','player1_button_B'],axis=1)


# In[50]:


# Define the mapping dictionary
mapping = {0: False, 1: True}

# Apply the mapping to each column in the dataset
data2 = data2.replace(mapping)


# In[51]:


data2


# In[52]:


data2


# In[53]:


X = data2.drop(['player2_button_up', 'player2_button_down', 'player2_button_left', 'player2_button_right',
               'player2_button_L', 'player2_button_R', 'player2_button_A', 'player2_button_X',
               'player2_button_B', 'player2_button_Y'], axis=1)


# In[54]:


y = data2[['player2_button_up', 'player2_button_down', 'player2_button_left', 'player2_button_right',
          'player2_button_L', 'player2_button_R', 'player2_button_A', 'player2_button_X',
          'player2_button_B', 'player2_button_Y']]


# In[55]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


# Create an instance of the StandardScaler
scalingDataset = StandardScaler()

# Fit and transform the training data
X_train = scalingDataset.fit_transform(X_train)

# Transform the testing data
X_test = scalingDataset.transform(X_test)


# In[57]:


input_shape = (X_train.shape[1],)

inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')(x)

fightingModel = tf.keras.Model(inputs=inputs, outputs=outputs)

fightingModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


fightingModel.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))


TotalLoss, Accuracy = fightingModel.evaluate(X_test, y_test)
print('Test Loss:', TotalLoss)
print('Test Accuracy:', Accuracy)


# In[58]:


predictedMoves = fightingModel.predict(X_test)


# In[59]:


predictedMoves[0]


# In[60]:


predictedMoves[1]


# In[61]:


predictedMoves[2]


# In[62]:


predictedButtons = (predictedMoves > 0.1)
predictedButtons = predictedButtons.astype(bool)


# In[63]:


for button in predictedButtons:
    print(button)


# In[64]:


fightingModel.save('ModelFile.h5')


# In[65]:


scaledFile = 'ScaledValues.save'
joblib.dump(scalingDataset, scaledFile)


# In[ ]:




