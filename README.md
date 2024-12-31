<h1>Predictive Model of Heart Failure </h1>

<p>This project aims to analyze and predict heart failure-related deaths using clinical and demographic data. The goal is to leverage machine learning models to identify high-risk patients and improve early intervention and patient care.</p>

---

<h2>Key Features</h2>
<ol>
    <li><strong>Exploratory Data Analysis (EDA) and Data Preprocessing:</strong>
    </li>
    <li><strong>Model Building and Comparison:</strong>
        <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
            <li>Support Vector Machine (SVM)</li>
        </ul>
    </li>
    <li><strong>Model Evaluation:</strong>
        <ul>
            <li>Comparison of models using ROC curves and AUC scores.</li>
            <li>Confusion Matrix</li>
        </ul>
    </li>
</ol>

---

<h2>Data Description</h2>
<p>The dataset includes the following key features:</p>
<ul>
    <li><strong>Clinical Features:</strong> Ejection fraction, serum creatinine, hypertension, etc.</li>
    <li><strong>Demographic Features:</strong> Age, gender, smoking status, etc.</li>
    <li><strong>Outcome:</strong> Death event (binary variable).</li>
</ul>

---

<h2>Visualizations</h2>
<h3>1. Ejection Fraction vs. Age by Death Event</h3>
<p>A scatter plot showcasing the relationship between ejection fraction and age, categorized by death event.</p>

![Query screencshot](https://github.com/sameena93/Death-rate-by-Cardiovascular-Disease/blob/main/img/ejection_fraction_by_age.jpeg)

<h3>2. Histogram of Serum Creatinine by Death Events with Hypertension</h3>
<p>A histogram to explore the distribution of serum creatinine levels, segmented by death events and hypertension status.</p>

![](https://github.com/sameena93/Death-rate-by-Cardiovascular-Disease/blob/main/img/creatbyhtn.jpeg)

<h3>3. Heatmap of Correlation Matrix</h3>
<p>A heatmap visualizing the correlation between various features in the dataset to identify significant relationships.</p>

![](https://github.com/sameena93/Death-rate-by-Cardiovascular-Disease/blob/main/img/heatmap%20of%20coorelation%20matrix.jpeg)

<h3>4. Death Event by Number of Hospital Visits</h3>
<p>A bar chart highlighting the frequency of death events based on the number of hospital visits.</p>

![](https://github.com/sameena93/Death-rate-by-Cardiovascular-Disease/blob/main/img/ejection%20fraction%20by%20age.jpeg)


<h3>5. Smoking Status by Gender</h3>
<p>A bar plot showing the distribution of smoking status for each gender.</p>

![](https://github.com/sameena93/Death-rate-by-Cardiovascular-Disease/blob/main/img/smokingbygender.jpeg)

<h2>Model Comparison and Evaluation</h2>
<p>The following models were implemented to predict the likelihood of heart failure-related deaths:</p>
<ul>
    <li>Logistic Regression</li>
    <li>Decision Tree</li>
    <li>Random Forest</li>
    <li>Support Vector Machine (SVM)</li>
</ul>

---

<h3>Evaluation Metrics:</h3>

![](https://github.com/sameena93/Death-rate-by-Cardiovascular-Disease/blob/main/img/ROC_plots.png) 

---

<h2>Conclusion</h2>
<p>The project demonstrates the effectiveness of machine learning models in predicting heart failure-related deaths. Based on the AUC scores and ROC curves, SVM emerged as the most accurate model. The insights from the visualizations and the models can aid in identifying high-risk patients and improving early intervention strategies.</p>

<h2>How to Run</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone &lt;repository_url&gt;</code></pre>
    </li>
 
    <li>View the visualizations and model comparison results in the output folder.</li>
</ol>



