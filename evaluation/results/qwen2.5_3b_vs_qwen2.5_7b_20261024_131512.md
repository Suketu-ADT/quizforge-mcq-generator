# LLM performance report

Generated 2026-10-24T13:15:12  
Options per question: 4  

## qwen2.5:3b

Questions: 40

| Metric | Value | Better |
|---|---:|---|
| Accuracy | 0.9250 | higher |
| Exact Match (EM) | 0.9250 | higher |
| Precision (macro) | 0.9268 | higher |
| Recall (macro) | 0.9222 | higher |
| F1-score (per class) | 0.9234 | higher |
| Macro-F1 | 0.9234 | higher |
| Micro-F1 | 0.9250 | higher |
| Weighted-F1 | 0.9261 | higher |
| Top-k accuracy | 1.0000 | higher |
| Matthews Correlation Coefficient (MCC) | 0.9006 | higher |
| Cohen's kappa | 0.8998 | higher |
| Log loss (cross-entropy) | 0.7401 | lower |
| Brier score | 0.1418 | lower |
| Expected Calibration Error (ECE) | 0.0706 | lower |

**Confusion matrix** (rows = correct option, columns = chosen option)

```
true \ predicted    A    B    C    D    ?
               A    8    1    0    0    0
               B    1    9    0    0    0
               C    0    1    9    0    0
               D    0    0    0   11    0
```

**Speed**

- Mean time per question = sum of latencies / N = 12.5s / 40 = 0.31s (model load time excluded; median 0.35s)
- Prompt processing speed = prompt_tokens / prompt_eval_seconds, mean = 824.6 tokens/s
- Generation speed = eval_count / (eval_duration / 1e9) = 128 / 9.93s = 12.9 tokens/s

<details><summary>Step-by-step calculations</summary>

### Accuracy
*Formula:* `correct predictions / total predictions`  

```
correct = 37, N = 40
Accuracy = 37 / 40 = 0.9250
95% Wilson confidence interval = [0.801, 0.974] (with only 40 questions the result is uncertain by about +/- 0.086)
For reference: random guessing among 4 options = 0.250
= 0.9250
```

### Exact Match (EM)
*Formula:* `predictions identical to the expected answer string / total`  

```
EM compares the normalised answer string with the expected one: 37 exact matches out of 40
EM = 37 / 40 = 0.9250
= 0.9250
```
*Answers here are single letters, so EM equals accuracy; unparseable replies count as non-matches.*

### Precision (macro)
*Formula:* `TP / (TP + FP), computed per option letter then averaged`  

```
Class A: TP=8 FP=1 FN=1  ->  P = 8/(8+1) = 0.8889, R = 8/(8+1) = 0.8889, F1 = 2*0.889*0.889/(0.889+0.889) = 0.8889
Class B: TP=9 FP=2 FN=1  ->  P = 9/(9+2) = 0.8182, R = 9/(9+1) = 0.9000, F1 = 2*0.818*0.900/(0.818+0.900) = 0.8571
Class C: TP=9 FP=0 FN=1  ->  P = 9/(9+0) = 1.0000, R = 9/(9+1) = 0.9000, F1 = 2*1.000*0.900/(1.000+0.900) = 0.9474
Class D: TP=11 FP=0 FN=0  ->  P = 11/(11+0) = 1.0000, R = 11/(11+0) = 1.0000, F1 = 2*1.000*1.000/(1.000+1.000) = 1.0000
Macro precision = (0.889 + 0.818 + 1.000 + 1.000) / 4 = 0.9268
= 0.9268
```

### Recall (macro)
*Formula:* `TP / (TP + FN), computed per option letter then averaged`  

```
Macro recall = (0.889 + 0.900 + 0.900 + 1.000) / 4 = 0.9222
= 0.9222
```

### F1-score (per class)
*Formula:* `2 * P * R / (P + R)`  

```
F1_A = 0.8889   (support = 9 questions)
F1_B = 0.8571   (support = 10 questions)
F1_C = 0.9474   (support = 10 questions)
F1_D = 1.0000   (support = 11 questions)
= 0.9234
```

### Macro-F1
*Formula:* `mean of per-class F1 (every class counts equally)`  

```
Macro-F1 = (0.889 + 0.857 + 0.947 + 1.000) / 4 = 0.9234
= 0.9234
```

### Micro-F1
*Formula:* `F1 from globally pooled TP, FP, FN`  

```
TP = 37, FP = 3, FN = 3 (summed over all classes)
P = 37/(37+3) = 0.9250, R = 37/(37+3) = 0.9250
Micro-F1 = 2*P*R/(P+R) = 0.9250
= 0.9250
```
*With one answer per question and no unparsed replies, micro-F1 equals accuracy.*

### Weighted-F1
*Formula:* `mean of per-class F1 weighted by support`  

```
Weighted-F1 = (0.889*9 + 0.857*10 + 0.947*10 + 1.000*11) / 40 = 0.9261
= 0.9261
```

### Top-k accuracy
*Formula:* `fraction of questions whose correct option is among the k most probable`  

```
Top-1 accuracy = 37 / 40 = 0.9250  (gold letter within the 1 most probable options)
Top-2 accuracy = 40 / 40 = 1.0000  (gold letter within the 2 most probable options)
Top-3 accuracy = 40 / 40 = 1.0000  (gold letter within the 3 most probable options)
= 1.0000
```
*Value shown is top-2. Top-1 equals accuracy.*

### Matthews Correlation Coefficient (MCC)
*Formula:* `(c*s - sum(p_k*t_k)) / sqrt((s^2 - sum p_k^2) * (s^2 - sum t_k^2))`  

```
s = N = 40, c = correctly predicted = 37
t_k (true counts)      = {'A': 9, 'B': 10, 'C': 10, 'D': 11}
p_k (predicted counts) = {'A': 9, 'B': 11, 'C': 9, 'D': 11}
sum(p_k*t_k) = 402, sum(p_k^2) = 404, sum(t_k^2) = 402
numerator = 37*40 - 402 = 1078
denominator = sqrt(1600-404) * sqrt(1600-402) = 1197.000
MCC = 0.9006   (1 = perfect, 0 = no better than chance, -1 = always wrong)
= 0.9006
```

### Cohen's kappa
*Formula:* `(p_o - p_e) / (1 - p_e)`  

```
p_o (observed agreement) = 37/40 = 0.9250
p_e (agreement expected by chance) = sum(p_k*t_k)/N^2 = 402/1600 = 0.2512
kappa = (0.9250 - 0.2512) / (1 - 0.2512) = 0.8998
= 0.8998
```

### Log loss (cross-entropy)
*Formula:* `-(1/N) * sum( ln p(correct option) )`  

```
per-question loss = -ln(p assigned to the correct option); first five: 0.000, 0.000, 0.000, 0.000, 0.000, ...
sum of 40 losses = 29.606
Log loss = 29.606 / 40 = 0.7401
For reference: always answering with equal probabilities = ln(4) = 1.386
= 0.7401
```
*Probabilities are clipped to at least 1e-15 so a confident wrong answer gives a large but finite penalty.*

### Brier score
*Formula:* `(1/N) * sum over questions of sum over options (p_option - y_option)^2`  

```
per-question score = squared error of the probability vector against the one-hot truth; first five: 0.000, 0.000, 0.000, 0.000, 0.000, ...
Brier = 5.673 / 40 = 0.1418
For reference: equal probabilities = 1 - 1/4 = 0.750; perfect = 0; worst = 2
= 0.1418
```

### Expected Calibration Error (ECE)
*Formula:* `sum over bins of (n_bin / N) * |accuracy_bin - confidence_bin|`  

```
confidence = probability of the predicted option; questions are grouped into 10 equal-width confidence bins
bin (0.8, 0.9]: n= 1  accuracy=0.000  confidence=0.891  |gap|=0.891  weight=1/40  contribution=0.0223
bin (0.9, 1.0]: n=39  accuracy=0.949  confidence=0.998  |gap|=0.050  weight=39/40  contribution=0.0483
ECE = 0.0706   (0 = confidence always matches how often the model is right)
= 0.0706
```
*A high-confidence model that is often wrong has a large ECE.*

</details>

## qwen2.5:7b

Questions: 40

| Metric | Value | Better |
|---|---:|---|
| Accuracy | 1.0000 | higher |
| Exact Match (EM) | 1.0000 | higher |
| Precision (macro) | 1.0000 | higher |
| Recall (macro) | 1.0000 | higher |
| F1-score (per class) | 1.0000 | higher |
| Macro-F1 | 1.0000 | higher |
| Micro-F1 | 1.0000 | higher |
| Weighted-F1 | 1.0000 | higher |
| Top-k accuracy | 1.0000 | higher |
| Matthews Correlation Coefficient (MCC) | 1.0000 | higher |
| Cohen's kappa | 1.0000 | higher |
| Log loss (cross-entropy) | 0.0059 | lower |
| Brier score | 0.0011 | lower |
| Expected Calibration Error (ECE) | 0.0055 | lower |

**Confusion matrix** (rows = correct option, columns = chosen option)

```
true \ predicted    A    B    C    D    ?
               A    9    0    0    0    0
               B    0   10    0    0    0
               C    0    0   10    0    0
               D    0    0    0   11    0
```

**Speed**

- Mean time per question = sum of latencies / N = 27.7s / 40 = 0.69s (model load time excluded; median 0.75s)
- Prompt processing speed = prompt_tokens / prompt_eval_seconds, mean = 376.3 tokens/s
- Generation speed = eval_count / (eval_duration / 1e9) = 128 / 22.28s = 5.7 tokens/s

<details><summary>Step-by-step calculations</summary>

### Accuracy
*Formula:* `correct predictions / total predictions`  

```
correct = 40, N = 40
Accuracy = 40 / 40 = 1.0000
95% Wilson confidence interval = [0.912, 1.000] (with only 40 questions the result is uncertain by about +/- 0.044)
For reference: random guessing among 4 options = 0.250
= 1.0000
```

### Exact Match (EM)
*Formula:* `predictions identical to the expected answer string / total`  

```
EM compares the normalised answer string with the expected one: 40 exact matches out of 40
EM = 40 / 40 = 1.0000
= 1.0000
```
*Answers here are single letters, so EM equals accuracy; unparseable replies count as non-matches.*

### Precision (macro)
*Formula:* `TP / (TP + FP), computed per option letter then averaged`  

```
Class A: TP=9 FP=0 FN=0  ->  P = 9/(9+0) = 1.0000, R = 9/(9+0) = 1.0000, F1 = 2*1.000*1.000/(1.000+1.000) = 1.0000
Class B: TP=10 FP=0 FN=0  ->  P = 10/(10+0) = 1.0000, R = 10/(10+0) = 1.0000, F1 = 2*1.000*1.000/(1.000+1.000) = 1.0000
Class C: TP=10 FP=0 FN=0  ->  P = 10/(10+0) = 1.0000, R = 10/(10+0) = 1.0000, F1 = 2*1.000*1.000/(1.000+1.000) = 1.0000
Class D: TP=11 FP=0 FN=0  ->  P = 11/(11+0) = 1.0000, R = 11/(11+0) = 1.0000, F1 = 2*1.000*1.000/(1.000+1.000) = 1.0000
Macro precision = (1.000 + 1.000 + 1.000 + 1.000) / 4 = 1.0000
= 1.0000
```

### Recall (macro)
*Formula:* `TP / (TP + FN), computed per option letter then averaged`  

```
Macro recall = (1.000 + 1.000 + 1.000 + 1.000) / 4 = 1.0000
= 1.0000
```

### F1-score (per class)
*Formula:* `2 * P * R / (P + R)`  

```
F1_A = 1.0000   (support = 9 questions)
F1_B = 1.0000   (support = 10 questions)
F1_C = 1.0000   (support = 10 questions)
F1_D = 1.0000   (support = 11 questions)
= 1.0000
```

### Macro-F1
*Formula:* `mean of per-class F1 (every class counts equally)`  

```
Macro-F1 = (1.000 + 1.000 + 1.000 + 1.000) / 4 = 1.0000
= 1.0000
```

### Micro-F1
*Formula:* `F1 from globally pooled TP, FP, FN`  

```
TP = 40, FP = 0, FN = 0 (summed over all classes)
P = 40/(40+0) = 1.0000, R = 40/(40+0) = 1.0000
Micro-F1 = 2*P*R/(P+R) = 1.0000
= 1.0000
```
*With one answer per question and no unparsed replies, micro-F1 equals accuracy.*

### Weighted-F1
*Formula:* `mean of per-class F1 weighted by support`  

```
Weighted-F1 = (1.000*9 + 1.000*10 + 1.000*10 + 1.000*11) / 40 = 1.0000
= 1.0000
```

### Top-k accuracy
*Formula:* `fraction of questions whose correct option is among the k most probable`  

```
Top-1 accuracy = 40 / 40 = 1.0000  (gold letter within the 1 most probable options)
Top-2 accuracy = 40 / 40 = 1.0000  (gold letter within the 2 most probable options)
Top-3 accuracy = 40 / 40 = 1.0000  (gold letter within the 3 most probable options)
= 1.0000
```
*Value shown is top-2. Top-1 equals accuracy.*

### Matthews Correlation Coefficient (MCC)
*Formula:* `(c*s - sum(p_k*t_k)) / sqrt((s^2 - sum p_k^2) * (s^2 - sum t_k^2))`  

```
s = N = 40, c = correctly predicted = 40
t_k (true counts)      = {'A': 9, 'B': 10, 'C': 10, 'D': 11}
p_k (predicted counts) = {'A': 9, 'B': 10, 'C': 10, 'D': 11}
sum(p_k*t_k) = 402, sum(p_k^2) = 402, sum(t_k^2) = 402
numerator = 40*40 - 402 = 1198
denominator = sqrt(1600-402) * sqrt(1600-402) = 1198.000
MCC = 1.0000   (1 = perfect, 0 = no better than chance, -1 = always wrong)
= 1.0000
```

### Cohen's kappa
*Formula:* `(p_o - p_e) / (1 - p_e)`  

```
p_o (observed agreement) = 40/40 = 1.0000
p_e (agreement expected by chance) = sum(p_k*t_k)/N^2 = 402/1600 = 0.2512
kappa = (1.0000 - 0.2512) / (1 - 0.2512) = 1.0000
= 1.0000
```

### Log loss (cross-entropy)
*Formula:* `-(1/N) * sum( ln p(correct option) )`  

```
per-question loss = -ln(p assigned to the correct option); first five: 0.000, 0.000, 0.000, 0.000, 0.078, ...
sum of 40 losses = 0.235
Log loss = 0.235 / 40 = 0.0059
For reference: always answering with equal probabilities = ln(4) = 1.386
= 0.0059
```
*Probabilities are clipped to at least 1e-15 so a confident wrong answer gives a large but finite penalty.*

### Brier score
*Formula:* `(1/N) * sum over questions of sum over options (p_option - y_option)^2`  

```
per-question score = squared error of the probability vector against the one-hot truth; first five: 0.000, 0.000, 0.000, 0.000, 0.011, ...
Brier = 0.045 / 40 = 0.0011
For reference: equal probabilities = 1 - 1/4 = 0.750; perfect = 0; worst = 2
= 0.0011
```

### Expected Calibration Error (ECE)
*Formula:* `sum over bins of (n_bin / N) * |accuracy_bin - confidence_bin|`  

```
confidence = probability of the predicted option; questions are grouped into 10 equal-width confidence bins
bin (0.8, 0.9]: n= 1  accuracy=1.000  confidence=0.855  |gap|=0.145  weight=1/40  contribution=0.0036
bin (0.9, 1.0]: n=39  accuracy=1.000  confidence=0.998  |gap|=0.002  weight=39/40  contribution=0.0019
ECE = 0.0055   (0 = confidence always matches how often the model is right)
= 0.0055
```
*A high-confidence model that is often wrong has a large ECE.*

</details>

## Comparison

```
Metric                                        qwen2.5:3b    qwen2.5:7b   Better
Accuracy                                          0.9250        1.0000   higher
Exact Match (EM)                                  0.9250        1.0000   higher
Precision (macro)                                 0.9268        1.0000   higher
Recall (macro)                                    0.9222        1.0000   higher
F1-score (per class)                              0.9234        1.0000   higher
Macro-F1                                          0.9234        1.0000   higher
Micro-F1                                          0.9250        1.0000   higher
Weighted-F1                                       0.9261        1.0000   higher
Top-k accuracy                                    1.0000        1.0000   higher
Matthews Correlation Coefficient (MCC)            0.9006        1.0000   higher
Cohen's kappa                                     0.8998        1.0000   higher
Log loss (cross-entropy)                          0.7401        0.0059   lower
Brier score                                       0.1418        0.0011   lower
Expected Calibration Error (ECE)                  0.0706        0.0055   lower
```
