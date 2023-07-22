### AutoML App using Streamlit, Pandas Profiling, and PyCaret

#### OVERVIEW

<i>This application allows users to build an automated Machine Learning (AutoML) pipeline with ease. The app is designed to streamline the process of data analysis, profiling, model building, and result downloading. It leverages the power of Streamlit, Pandas Profiling, and PyCaret to provide a user-friendly and efficient experience.</i>

#### FEATURES

##### 1. UPLOAD
<i>In the ***Upload*** section, users can upload their dataset to get started with the AutoML pipeline. The app will automatically detect the data types and preview the dataset for further analysis.</i>

##### 2. PROFILING
<i>The ***Profiling*** section utilizes the power of Pandas Profiling to generate a comprehensive report on the uploaded dataset. Users can get insights into data distributions, missing values, correlations, and more. This profiling report helps users better understand the data and make informed decisions during the modeling process.</i>

##### 3. MODELLING
<i>The ***Modelling*** section is where the magic happens! Users can select the target variable. The app will automatically preprocess the data, split it into training and testing sets, and apply PyCaret's AutoML functionality. Users can choose to run the AutoML process with a specific time limit or until the best model is found. The app will pick the top-performing model.</i>

##### 4. DOWNLOAD
<i>In the ***Download*** section, users can download the best-performing model. This enables users to deploy the model or share the results with others seamlessly.</i>

#### HOW TO USE
<i>
  
1. Clone this repository to your local machine.  
2. Install the required dependencies using ***pip install -r requirements.txt***. 
3. Run the app using ***streamlit run app.py***. 
4. Once the app is launched, you'll see the navigation menu on the sidebar.
5. Start by clicking on the Upload option to upload your dataset.
6. Move on to the ***Profiling*** section to get insights into your data.
7. Proceed to the ***Modeling*** section to build and evaluate your ML models with PyCaret.
8. Finally, head to the ***Download*** section to save the best model.
</i>

### REFERENCES

<i>Youtube and Github</i>

<i>Please feel free to open issues or pull requests on my GitHub repository.</i>


**Happy AutoML-ing!**
