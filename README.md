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

### assesment.csv 
* `code_module` {PPK} // partial FK to courses
  * Identification code of the module, to which the assessment belongs.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `code_presentation` {PPK} // partial FK to courses
   * Identification code of the presentation, to which the assessment belongs.
   * CHAR(5), categorical with levels '2013B', '2013J', '2014B', '2014J'
* `id_assessment` {PK}
   * Identification number of the assessment.
   * INT, from 1752 to 40088
* `assessment_type`
   * Type of assessment. Three types of assessments exist: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA) and Final Exam (Exam).
   * VARCHAR(4), categorical with levels 'TMA', 'CMA', 'Exam'
* `date`
   * Information about the final submission date of the assessment calculated as the number of days since the start of the module-presentation. The starting date of the presentation has number 0 (zero).
  * INT, from 12 to 261
* `weight`
  * Weight of the assessment in %. Typically, Exams are treated separately and have the weight 100%; the sum of all other assessments is 100%.
  * INT, 0 to 100

### courses.csv 
* `code_module` {PPK}
  * code name of the module (known as unit code in many universities), which serves as the identifier.
  * CHAR(3), categorical with levels 'BBB', 'DDD', 'EEE', 'FFF', 'GGG'
* `code_presentation` {PPK}
  * Code name of the presentation. It consists of the year and “B” for the presentation starting in February (Corresponds to Semester2) and “J” for the presentation starting in October(Corresponds to Semester1).
  * CHAR(5), categorical with levels '2013B', '2013J', '2014B', '2014J'
* `module_presentation_length`
  * Length of the module-presentation in days.
  * INT, from 234 to 269

### studentAssesment.csv 
* `id_assessment` {PPK} // FK to assessment
  * Identification number of the assessment.
  * INT, from 1752 to 37443
* `id_student` {PPK}
  * A unique identification number for the student.
  * INT, from 6516 to 2698588
* `date_submitted`
  * The date of student submission, measured as the number of days since the start of the module presentation.
  * INT, from -11 to 608
* `is_banked`
  * A status flag indicating that the assessment result has been transferred from a previous presentation.
  * INT, (0,1) standing in for true false
* `score`
  * The student’s score in this assessment. The range is from 0 to 100. The score lower than 40 is interpreted as Fail. The marks are in the range from 0 to 100.
  * INT, from 0 to 100

### studentInfo.csv 
* `code_module` {PPK} // partial FK to courses
  * Identification code of the module, to which the assessment belongs.
  * CHAR(3), categorical with levels 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `code_presentation` {PPK} // partial FK to courses
  * Identification code of the presentation, to which the assessment belongs.
  * CHAR(5), categorical with levels '2013B', '2013J', '2014B', '2014J'
* `id_student` {PPK}
  * A unique identification number for the student.
  * INT, from 3733 to 2716795
* `gender` 
  * The student’s gender.
  * CHAR(1), categorical with levels 'M', 'F'
* `region` 
  * Identifies the geographic region, where the student lived while taking the module  *presentation.
  * VARCHAR(45), categorical with levels 'Scotland', 'East Anglian Region', 'London Region', 'South Region', 'North Western Region'
* `highest_education` 
  * Highest student education level on entry to the module presentation.
  * VARCHAR(45), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `imd_band` 
  * Specifies the Index of Multiple Depravation band of the place where the student lived during the module  *presentation.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `c` 
  * I.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `c` 
  * I.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `c` 
  * I.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `c` 
  * I.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'
* `c` 
  * I.
  * CHAR(3), categorical with levels 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'

