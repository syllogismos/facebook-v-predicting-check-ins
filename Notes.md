# Dialy Notes Log

## 20th May

#### MAP@3 :
    Understanding mean average precision MAP@3. Generally used in evaluating ranked retrieval results, in Information Retrieval.
    Helpful link: http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
    
    We use precision, recall, F value to evaluate where rank isn't really relevant.
    
    Say for example. in a set of [a,b,c,d,e] actual true values are [a,b,c] and the predicted true values are [a,c,d]
    
    True positives(tp): a,c
    False positives(fp): d
    True negatives(tn): e
    False negatives(fn): b
    
    Precision: How precise is our prediction. (true positives)/(no of predictions)
        tp/(tp + fp)
    Recall: What percent of true values are returned. (true positives)/(all true values)
        tp/(tp + fn)
    Accuracy: Total number of correct predictions. (true positives + true negatives)/everything
        (tp + tn)/(tp + fp + tn + fn)
    
    In the above example:
        Precision = [a,c]/[a,c,d] = 2/3
        Recall = [a,c]/[a,b,c] = 2/3 
        Accuracy = [a,c] + [e]/[a,b,c,d,e] = 3/5
        
    * As we make more and more predictions, if there are no new true positives, the recall stays constant, but precision drops.
    * Recall means out of all the positives, how many did we predict correctly.
    * Precision means out of all our predictions, how many did we get it right.
    * Recall is 1 if we return everything as a prediction. But the precicion drops significantly.
    * The reason why accuracy isn't a good measure is cause when the data is highly skewed, it makes no sense. Say for example only 0.01% of values will be true. You can just predict everything to be false still end up with 99.99 accuracy.
    * However, labeling all documents as nonrelevant is completely unsatisfying to an information retrieval system user. Users are always going to want to see some documents, and can be assumed to have a certain tolerance for seeing some false positives providing that they get some useful information. The measures of precision and recall concentrate the evaluation on the return of true positives, asking what percentage of the relevant documents have been found and how many false positives have also been returned.
    * The advantage of having the two numbers for precision and recall is that one is more important than the other in many circumstances. 
    * Typical web surfers would like every result on the first page to be relevant (high precision) but have not the slightest interest in knowing let alone looking at every document that is relevant. 
    * In contrast, various professional searchers such as paralegals and intelligence analysts are very concerned with trying to get as high recall as possible, and will tolerate fairly low precision results in order to get it. Individuals searching their hard disks are also often interested in high recall searches. 
    * Nevertheless, the two quantities clearly trade off against one another: you can always get a recall of 1 (but very low precision) by retrieving all documents for all queries! Recall is a non-decreasing function of the number of documents retrieved. 
    * On the other hand, in a good system, precision usually decreases as the number of documents retrieved is increased. 
    * In general we want to get some amount of recall while tolerating only a certain percentage of false positives.
    * A single measure that trades off precision versus recall is the F measure , which is the weighted harmonic mean of precision and recall:
    
    F = 1/[(alpha * 1/P) + ((1-alpha) * 1/R)] = (beta^2 + 1)PR/[beta^2*P + R] where beta = (1 - alpha)/alpha
    alpha belongs to [0,1] and beta^2 belongs to [0, inf]
    
    F is weighted harmonic mean of precision and recall, F1 is when beta = 1 and a simple harmonic mean of precision and recall
    
    beta < 1 emphasizes precision, beta > 1 emphasizes recall.