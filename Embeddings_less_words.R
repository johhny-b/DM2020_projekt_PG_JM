options(width = 120)
library(tm)
library(data.table)
library(wordVectors)
library(glmnet)
library(foreach)
library(doParallel)

### Najpierw wczytujê dane i robiê czyszczenie.
# W labie 10 brakowa³o usuwania interpunkcji!
# Przez to "students" oraz "students'" to by³y dwa ró¿ne s³owa
# Dodatkowo, usuwam stopwords oraz robiê stemming (nie wiem czy to polski odpowiednik?)

path = 'DM2020_training_docs_and_labels.txt' #œcie¿ka do pliku
docs = data.table::fread(path, header = FALSE, sep = "\t", encoding = "UTF-8")
setnames(docs, c("doc_id", "text", "labels"))
myCorpus = Corpus(DataframeSource(docs),
                  readerControl = list(reader = readDataframe, language = "en"))
# you can always add a new one
correctEnc = function(x) {
  stringr::str_replace_all(x,"[^[:graph:]]", " ")
}

# using custom transformers
myCorpus = tm_map(myCorpus, content_transformer(correctEnc))
myCorpus = tm_map(myCorpus, content_transformer(tolower))

# using 'standard' transformers (one-by-one)
myCorpus = tm_map(myCorpus, removeNumbers)
myCorpus = tm_map(myCorpus, removePunctuation)
myCorpus = tm_map(myCorpus, removeWords, stopwords())
myCorpus = tm_map(myCorpus, stemDocument)
myCorpus = tm_map(myCorpus, stripWhitespace)

docs2 = data.frame(doc_id = docs[,doc_id], 
                   text = sapply(myCorpus, as.character), stringsAsFactors = FALSE,
                   labels = docs[,labels])
docs2 = as.data.table(docs2)

docs2[, text := gsub("[<][^<>]+[>]", "", text)]
docs2[, text := gsub("[^[:alpha:]\']+", " ", text)]
docs2[, text := gsub("[[:punct:][:digit:][:space:]]+", " ", text)]

writeLines(docs2[, text], "new_docs_for_training.txt")
word_vectors_cbow = train_word2vec("new_docs_for_training.txt",
                                   "words_skipgram_size50.bin",
                                   vectors=100, threads=6, window=5, iter=20, 
                                   negative_samples=5, cbow = 0, force = TRUE)
save(word_vectors_skipgram,file="new_word_vectors_skipgram.RData")

doc_embeddings = sapply(strsplit(docs[, text], " "), 
                        function(x, word_embedds) word_embedds[[x]],
                        word_vectors_skipgram)
doc_embeddings = t(doc_embeddings)


#Teraz mo¿na dokonywaæ predykcji. Najpierw trzeba przygotowaæ labele:
labels = strsplit(docs[, labels], ",")
labels = data.table(doc_id = rep(1:length(labels), times = sapply(labels, length)),
                    label = unlist(labels))
labels[, value := 1]
labels = dcast(labels, doc_id ~ label, value.var = "value", fill = 0)
labels[, doc_id := NULL]
labels[, colnames(labels)[colSums(labels) < 100] := NULL]
dim(labels)

#Dzielimy na zbiór testowy i walidacyjny (w stosunku 9 do 1). Trzeba pamiêtaæ, ¿eby potem wytrenowaæ na ca³ym zbiorze
val_idx = sort(sample(nrow(doc_embeddings), 10000))
# example: elastic regression
registerDoParallel(5)
model = glmnet::cv.glmnet(doc_embeddings[-val_idx, ], as.matrix(labels)[-val_idx, ], 
                          family = 'multinomial', 
                          nfolds = 5, alpha = 0.0, type.measure="mse", 
                          nlambda = 100, parallel = TRUE, maxit=10000)

# validation score:
preds = predict(model, doc_embeddings[val_idx, ], 
                type="response", s=model$lambda.min)
predicted_labels = apply(preds, c(1,3), function(x) names(which(x > 0.5*max(x))))
# our evaluation metric:
F1score = function(x, y) {
  common = intersect(x, y)
  result = 0
  if(length(common) == 0) {
    result = 0
  } else {
    result = 2*(length(common)^2)/(length(x)*length(y))/(length(common)/length(x) + length(common)/length(y))
  }
  
  as.numeric(result)
}
true_labels = strsplit(docs[, labels], ",")[val_idx]
mean(mapply(F1score, predicted_labels, true_labels))

docs2 = VCorpus(DirSource('DM2020_training_docs_and_labels.txt'),
                readerControl = list(reader = readPlain, language = "en"))
