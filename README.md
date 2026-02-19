# Places Attribute Conflation  
**Project A – Winter 2026**  
**Author:** Karthik Chaparala

---

## Overview

This project focuses on building a clean and reliable golden record for a real-world place when the same place appears in multiple data sources with conflicting or inconsistent information.

In real-world places data, it is common to have one physical place with multiple data sources and slightly different (and sometimes incorrect or outdated) attributes.

The goal of this project is to automatically decide which attributes should be trusted and kept.

---


## Project Goals

The main goals of this project are to:

1. Build a high-quality labeled gold dataset from pre-matched place pairs. The golden dataset will have a "golden" value for the following attributes:
- Names
- Categories
- Websites
- Socials
- Emails
- Phone Numbers
- Brands
- Addresses
The gold value for each attribute is the one that is most accurate in the pair given.

2. Design an algorithm that can select the best attribute values when multiple candidates exist.
3. Compare rule-based and machine-learning approaches.
4. Evaluate the quality of the selected attributes using clear metrics.

---
