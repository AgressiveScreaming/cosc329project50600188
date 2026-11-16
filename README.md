# COSC 329 - Learning Analytics Project

## Assignment
Predict "at-risk" students after an early assessment using the Open University Learning Analytics Dataset (OULAD). Explore the dataset to identify courses, students, and their course-level data (engagement with course material, interaction in discussion forums, and assessment data). 
Massage the data into feature vectors that would be appropriate for classification models. Be careful how you engineer these features because that will impact your performance in the end, so document and discuss tradeoffs you make in this step. 
Classify your data using random forest, XGBoost, logistic regression, support vector machines, and naive Bayes. When you train the machine learning model, you will notice that most students pass and the model will therefore most likely predict a pass for all students. So use a synthetic minority over-sampling technique (SMOTE) as a precaution. The evaluation metrics include precision, recall, F1, and ROC-AUC score for all your classification algorithms.

Transform the data into features (document choices and tradeoffs), simulate the midterm "cut-off" to predict final course grade, use 10-fold cross-validation and different splits to train and test your data for each classification algorithm (remember to use SMOTE for sample selection during training but not testing), use binary classification for students at risk of failing or dropping out, repeat this 10-fold cross validation across 25 independent trials with different random shuffles, report on average and standard deviations on 4 metrics above, include explanations and meaningful interpretation that reflects the type of data used and how the algorithms work.

## Current plan/timeline
- Week 9: Data setup - current week
- Week 10: logistic regression, naive Bayes
- Week 11: random forest, XGBoost,
- Week 12: SVM and start presentation
- Week 13: Submit Presentation
- Week 14: Submit Deliverables

## Data Structure - taken/modified from the OULAD dataset website
<img width="1410" height="1330" alt="image" src="https://github.com/user-attachments/assets/1ff42445-bd58-4b7c-975d-5fb3b5c3b86f" />

```
Assesments
    code_module code_presentation  id_assessment assessment_type   date  weight
0           AAA             2013J           1752             TMA   19.0    10.0
1           AAA             2013J           1753             TMA   54.0    20.0
2           AAA             2013J           1754             TMA  117.0    20.0
3           AAA             2013J           1755             TMA  166.0    20.0
4           AAA             2013J           1756             TMA  215.0    30.0
..          ...               ...            ...             ...    ...     ...
201         GGG             2014J          37443             CMA  229.0     0.0
202         GGG             2014J          37435             TMA   61.0     0.0
203         GGG             2014J          37436             TMA  124.0     0.0
204         GGG             2014J          37437             TMA  173.0     0.0
205         GGG             2014J          37444            Exam  229.0   100.0

[206 rows x 6 columns]


Courses
   code_module code_presentation  module_presentation_length
0          AAA             2013J                         268
1          AAA             2014J                         269
2          BBB             2013J                         268
3          BBB             2014J                         262
4          BBB             2013B                         240
5          BBB             2014B                         234
6          CCC             2014J                         269
7          CCC             2014B                         241
8          DDD             2013J                         261
9          DDD             2014J                         262
10         DDD             2013B                         240
11         DDD             2014B                         241
12         EEE             2013J                         268
13         EEE             2014J                         269
14         EEE             2014B                         241
15         FFF             2013J                         268
16         FFF             2014J                         269
17         FFF             2013B                         240
18         FFF             2014B                         241
19         GGG             2013J                         261
20         GGG             2014J                         269
21         GGG             2014B                         241


studentAssessment
        id_assessment  id_student  date_submitted  is_banked  score
0                1752       11391              18          0   78.0
1                1752       28400              22          0   70.0
2                1752       31604              17          0   72.0
3                1752       32885              26          0   69.0
4                1752       38053              19          0   79.0
...               ...         ...             ...        ...    ...
173907          37443      527538             227          0   60.0
173908          37443      534672             229          0  100.0
173909          37443      546286             215          0   80.0
173910          37443      546724             230          0  100.0
173911          37443      558486             224          0   80.0

[173912 rows x 5 columns]


StudentInfo
      code_module code_presentation  ... disability final_result
0             AAA             2013J  ...          N         Pass
1             AAA             2013J  ...          N         Pass
2             AAA             2013J  ...          Y    Withdrawn
3             AAA             2013J  ...          N         Pass
4             AAA             2013J  ...          N         Pass
...           ...               ...  ...        ...          ...
32588         GGG             2014J  ...          N         Fail
32589         GGG             2014J  ...          N  Distinction
32590         GGG             2014J  ...          Y         Pass
32591         GGG             2014J  ...          N    Withdrawn
32592         GGG             2014J  ...          N  Distinction

[32593 rows x 12 columns]


StudentRegistration
      code_module  ...  date_unregistration
0             AAA  ...                  NaN
1             AAA  ...                  NaN
2             AAA  ...                 12.0
3             AAA  ...                  NaN
4             AAA  ...                  NaN
...           ...  ...                  ...
32588         GGG  ...                  NaN
32589         GGG  ...                  NaN
32590         GGG  ...                  NaN
32591         GGG  ...                101.0
32592         GGG  ...                  NaN

[32593 rows x 5 columns]


StudentVle
         code_module code_presentation  id_student  id_site  date  sum_click
0                AAA             2013J       28400   546652   -10          4
1                AAA             2013J       28400   546652   -10          1
2                AAA             2013J       28400   546652   -10          1
3                AAA             2013J       28400   546614   -10         11
4                AAA             2013J       28400   546714   -10          1
...              ...               ...         ...      ...   ...        ...
10655275         GGG             2014J      675811   896943   269          3
10655276         GGG             2014J      675578   896943   269          1
10655277         GGG             2014J      654064   896943   269          3
10655278         GGG             2014J      654064   896939   269          1
10655279         GGG             2014J      654064   896939   269          1

[10655280 rows x 6 columns]


Vle
      id_site code_module code_presentation activity_type  week_from  week_to
0      546943         AAA             2013J      resource        NaN      NaN
1      546712         AAA             2013J     oucontent        NaN      NaN
2      546998         AAA             2013J      resource        NaN      NaN
3      546888         AAA             2013J           url        NaN      NaN
4      547035         AAA             2013J      resource        NaN      NaN
...       ...         ...               ...           ...        ...      ...
6359   897063         GGG             2014J      resource        NaN      NaN
6360   897109         GGG             2014J      resource        NaN      NaN
6361   896965         GGG             2014J     oucontent        NaN      NaN
6362   897060         GGG             2014J      resource        NaN      NaN
6363   897100         GGG             2014J      resource        NaN      NaN

[6364 rows x 6 columns]
```
