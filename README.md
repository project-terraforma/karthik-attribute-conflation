# Places Attribute Conflation  
**Project A – Winter 2026**  
**Author:** Karthik Chaparala

---

## Overview

Real-world place datasets often contain multiple records describing the same physical location collected from different data sources. These sources may contain conflicting, incomplete, or outdated attribute values.

The goal of this project is to determine **which attribute values should be trusted when multiple candidate values exist for the same place**.

This project focuses on **attribute conflation**, the process of selecting the most reliable attribute value from multiple sources to create a clean and consistent representation of a place.

---

## Project Goals

The main goals of this project are to:

### 1. Construct a Golden Dataset

A manually labeled **gold dataset** was created from pre-matched place record pairs.  
Each row represents two records believed to refer to the same place.

For each attribute, a label identifies which value should be considered the correct one.

The evaluated attributes include:

- **Names**
- **Categories**
- **Websites**
- **Socials**
- **Emails**
- **Phone Numbers**
- **Brands**
- **Addresses**

Each attribute is labeled using one of four possible outcomes:

| Label | Meaning |
|------|--------|
| **L** | Left value is better |
| **R** | Right value is better |
| **B** | Both values are equivalent |
| **N** | Neither value is useful |

The golden dataset represents **human judgment of the best attribute value within each pair**.

---

### 2. Design an Attribute Conflation Method

A **rule-based conflation algorithm** was implemented to automatically select the best attribute value.

The method evaluates candidate values using attribute-specific heuristics such as:

- completeness of the value
- formatting quality
- normalization and similarity comparisons
- URL and phone structure
- address component completeness

When two values are similar in quality, the algorithm applies **source reliability and confidence scores as tie-breakers**.

---

### 3. Evaluate Algorithm Performance

The algorithm's predictions are compared against the manually labeled golden dataset.

Performance is evaluated using two metrics:

#### Accuracy
The percentage of predictions that match the gold labels.

#### Macro F1 Score
A balanced evaluation metric that considers performance across all decision categories:

- Left (L)
- Right (R)
- Both (B)
- Neither (N)

Macro F1 combines **precision and recall** for each class and averages them to provide a robust measure of prediction quality.

---

## Method Overview

The conflation algorithm follows these steps:

1. Extract attribute values from each place pair.
2. Normalize values for comparison (e.g., URL canonicalization, phone normalization).
3. Apply attribute-specific scoring rules to evaluate value quality.
4. Select the better value or mark both/neither when appropriate.
5. Compare predictions against the golden dataset to compute evaluation metrics.

---

## Dataset

The project uses a dataset of **pre-matched place record pairs** provided in `project_a_samples.parquet`.

Each row contains two records:

- a **base record**
- a **candidate record**

The dataset includes attributes such as:

- names
- categories
- contact information
- brand information
- addresses
- data source metadata
- confidence scores

The manually labeled golden dataset is derived from this source data.

---

## Evaluation

The algorithm's predictions are evaluated against the golden dataset using:

- **Accuracy**
- **Macro F1 Score**

These metrics measure how closely the automated conflation decisions match human-labeled ground truth.

---

## Key Challenges

Several challenges were encountered during the project:

- inconsistent formatting across data sources
- empty structured objects representing missing values
- shortened URLs and redirect links
- address formatting differences across regions
- distinguishing between equivalent and slightly better values

These challenges highlight the complexity of real-world data integration tasks.

---

## Conclusion

This project demonstrates a rule-based approach to **place attribute conflation** using manually labeled training data and structured heuristics.

The method provides a baseline for automatically selecting reliable attribute values from multiple sources and can serve as a foundation for future improvements using machine learning or semantic similarity models.
