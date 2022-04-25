### Decision Tree

![Decision Tree](https://github.com/eclipse7723/machine_learning/blob/master/lab2/output/example/graphviz_tree_gini.png?raw=True)

### Features importances

![Decision Tree](https://github.com/eclipse7723/machine_learning/blob/master/lab2/output/example/gini_feature_importances_bar.png?raw=True)

### Accuracy affects

![Accuracy affects](https://github.com/eclipse7723/machine_learning/blob/master/lab2/output/example/accuracy_affects.png?raw=True)

### Multiclass Confusion Matrix

![Multiclass Confusion Matrix of gini_test_sample](https://github.com/eclipse7723/machine_learning/blob/master/lab2/output/example/gini_test_confusion_matrix.png?raw=true)

<table>
    <tr>
        <th>id</th>
        <th>label</th>
    </tr>
    	<tr>
		<td>0</td>
		<td>Normal_Weight</td>
	</tr>
	<tr>
		<td>1</td>
		<td>Overweight_Level_I</td>
	</tr>
	<tr>
		<td>2</td>
		<td>Overweight_Level_II</td>
	</tr>
	<tr>
		<td>3</td>
		<td>Obesity_Type_I</td>
	</tr>
	<tr>
		<td>4</td>
		<td>Insufficient_Weight</td>
	</tr>
	<tr>
		<td>5</td>
		<td>Obesity_Type_II</td>
	</tr>
	<tr>
		<td>6</td>
		<td>Obesity_Type_III</td>
	</tr>
</table>

##### How understand TP, TN, FP, FN from this matrix?

* **TP** _(True Positive)_: The actual value and predicted value should be the same.
For class 0 (Normal_Weight) we have `TP = 57`.

* **FN** _(False Negative)_: The sum of values of corresponding rows except the TP value.
For class 0 (Normal_Weight) we have `FN = 18 + 5 + 0 + 5 + 0 + 0 = 28`.

* **FP** _(False Positive)_: The sum of values of corresponding column except the TP value.
For class 0 (Normal_Weight) we have `FP = 3 + 0 + 0 + 3 + 0 + 0 = 6`.

* **TN** _(True Negative)_: The sum of values of all columns and row except the values of that class
that we are calculating the values for.
For class 0 (Normal_Weight) we have `TN = 57 + 37 + 1 + 0*3 + ... + 0*2 + 1 + 0 + 104 = 542`.

[source](https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/#:~:text=Confusion%20Matrix%20is%20used%20to,number%20of%20classes%20or%20outputs.)
