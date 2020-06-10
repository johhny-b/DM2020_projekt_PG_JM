options(width = 120)
library(tm)
library(data.table)
library(wordVectors)
library(glmnet)
library(foreach)
library(doParallel)

docs = data.table::fread('DM2020_training_docs_and_labels.txt', header = FALSE, sep = "\t", encoding = "UTF-8")

setnames(docs, c("doc_id", "text", "labels"))
# we need to do some basic text cleaning but typically, we do not stem words and remove stop-words
docs[, text := gsub("[<][^<>]+[>]", "", text)]
docs[, text := gsub("[^[:alpha:]\']+", " ", text)]
docs[, text := tolower(text)]
# this method is using a streaming API so we can write the documents to disc and save memory
writeLines(docs[, text], "docs_for_training.txt")

#W poni¿szej funkcji argument vectors oznacza wymiar przestrzeni w której chcemy reprezentowaæ s³owa
#Mo¿na siê zastanowiæ, czy vectors=100 jest optymalne
#Poni¿sza funkcja koduje s³owa (nie dokumenty!) jako wektory 
word_vectors_cbow = train_word2vec("docs_for_training.txt",
                                   "words_cbow_size50.bin",
                                   vectors=100, threads=6, window=5, iter=20, 
                                   negative_samples=5, cbow = 1, force = TRUE)
save(word_vectors_cbow,file="word_vectors_cbow.RData")

#Inna metoda: skipgram
word_vectors_skipgram = train_word2vec("docs_for_training.txt",
                                       "words_skipgram_size50.bin",
                                       vectors=100, threads=6, window=10, iter=10, 
                                       negative_samples=5, cbow = 0, force = TRUE)
save(word_vectors_skipgram,file="word_vectors_skipgram.RData")


#Teraz robiê embedding dla wszystkich dokumentów. Pan Janusz wspomina, ¿e mo¿na to zrobiæ sprytniej
#Jeœli bêdziemy mieli s³abe wyniki, to mo¿na siê zastanowiæ nad tym punktem
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
