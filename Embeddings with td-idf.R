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

path = 'DM2020_training_docs_and_labels.txt' #œcie¿ka do pliku
docs = data.table::fread(path, header = FALSE, sep = "\t", encoding = "UTF-8")

setnames(docs, c("doc_id", "text", "labels"))
# we need to do some basic text cleaning but typically, we do not stem words and remove stop-words
docs[, text := gsub("[<][^<>]+[>]", "", text)]
docs[, text := gsub("[^[:alpha:]\']+", " ", text)]
docs[, text := gsub("[[:punct:][:digit:][:space:]]+", " ", text)]
docs[, text := tolower(text)]

#Teraz trzeba jeszcze raz wytrenowaæ skipgram
# this method is using a streaming API so we can write the documents to disc and save memory
writeLines(docs[, text], "docs_for_training.txt")

word_vectors_skipgram = train_word2vec("docs_for_training.txt",
                                       "words_skipgram_size50.bin",
                                       vectors=100, threads=6, window=10, iter=10, 
                                       negative_samples=5, cbow = 0, force = TRUE)
save(word_vectors_skipgram,file="word_vectors_skipgram.RData")

### Teraz zamieniam docs na corpus, ¿eby u¿yæ funkcji z pakietu tm do policzenia tf-idf
# Próbowa³em wczeœniej liczyæ to rêcznie, ale siê liczy³o kilka godzin i siê nie doliczy³o
# a w pakiecie tm siê natychmiast
myCorpus = Corpus(DataframeSource(docs),
                   readerControl = list(reader = readDataframe, language = "en"))

DTM = DocumentTermMatrix(myCorpus, 
                          control = list(weighting = weightTfIdf))
# Jest tylko jeden problem - DTM zawiera mniej s³ów ni¿ nasz word_vectors_skipgram.
# Trzeba znaleŸæ te dodatkowe s³owa i usun¹æ z docs

wordsDTM = colnames(DTM) #To s¹ wszystkie s³owa wystêpuj¹ce w DTM
wordsSkipgram = rownames(word_vectors_skipgram) #To s¹ wszystkie s³owa wystêpuj¹ce w word_vectors_skipgram

length(wordsDTM)
length(wordsSkipgram)

# S³owa które s¹ w Skipgram, a nie ma ich w DTM
missing_words = lapply(wordsSkipgram,function(x, words) if(!(x%in%words)){return(x)},wordsDTM)
missing_words = missing_words[-which(sapply(missing_words, is.null))]
# W missing_words s¹ g³ównie dwuliterowe s³owa, oraz kilka pojedyñczych liter

#S³owa które s¹ w DTM, a nie ma ich w Skipgram
missing_words2 = lapply(wordsDTM,function(x, words) if(!(x%in%words)){return(x)},wordsSkipgram)
missing_words2 = missing_words2[-which(sapply(missing_words2, is.null))]
# S¹ tu prawie 93 tysi¹ce s³ów. Dlaczego nie ma ich w word_vectors_skipgram? Nie mam pojêcia.
# Byæ mo¿e dlatego, ¿e wystêpowa³y za rzadko.
# Natomiast "nadmiarowe" missing_words2 nie s¹ problemem (nie muszê siê do nich odwo³ywaæ)
# Problemem s¹ missing_words, które muszê usun¹æ z docs

tmp = sapply(docs[,text], function(x, missing) removeWords(unlist(x), missing), missing_words)
#Uwaga: powy¿sze na razie nie dzia³a.


###Dalsza idea by³a taka, ¿eby wzi¹æ ka¿dy dokument zakodowaæ w sposób:
#  - ka¿de s³owo zakodowaæ odpowiadaj¹cym mu wektorem z word_vectors_skipgram
#  - ka¿dy z tych wektorów pomno¿yæ przez tf-idf tego s³owa w tym dokumencie.
#  - zsumowaæ
# Wygl¹da³oby to mniej wiêcej tak, ale nie wiem, czy dzia³a, bo nie dotar³em do tego momentu
# Wczeœniej, poniewa¿ skipgram i DTM mia³y inne zbiory s³ów, to wywala³a siê linijka inspect(DTM[i,row.names(wordVec)])
doc_embeddings = matrix( rep( 0, len=100*100000), nrow = 100000)
for(i in 1:nrows(DTM)){
  row = strsplit(docs[i, text], " ")[[1]]
  wordVec = word_vectors_skipgram[[row, average=F]] #wordVec to macierz 100 na (liczba s³ów w dokumencie)
  wordVec = wordVec * inspect(DTM[i,row.names(wordVec)])
  doc_embeddings[i,] = apply(wordVec, 2, sum)
}

