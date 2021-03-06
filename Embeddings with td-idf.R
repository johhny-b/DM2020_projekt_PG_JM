options(width = 120)
library(tm)
library(data.table)
library(wordVectors)
library(glmnet)
library(foreach)
library(doParallel)

### Najpierw wczytuj� dane i robi� czyszczenie.
# W labie 10 brakowa�o usuwania interpunkcji!
# Przez to "students" oraz "students'" to by�y dwa r�ne s�owa

path = 'DM2020_training_docs_and_labels.txt' #�cie�ka do pliku
docs = data.table::fread(path, header = FALSE, sep = "\t", encoding = "UTF-8")

setnames(docs, c("doc_id", "text", "labels"))
# we need to do some basic text cleaning but typically, we do not stem words and remove stop-words
docs[, text := gsub("[<][^<>]+[>]", "", text)]
docs[, text := gsub("[^[:alpha:]\']+", " ", text)]
docs[, text := gsub("[[:punct:][:digit:][:space:]]+", " ", text)]
docs[, text := tolower(text)]

#Teraz trzeba jeszcze raz wytrenowa� skipgram
# this method is using a streaming API so we can write the documents to disc and save memory
writeLines(docs[, text], "docs_for_training.txt")

word_vectors_skipgram = train_word2vec("docs_for_training.txt",
                                       "words_skipgram_size50.bin",
                                       vectors=100, threads=6, window=10, iter=10, 
                                       negative_samples=5, cbow = 0, force = TRUE)
save(word_vectors_skipgram,file="word_vectors_skipgram.RData")

### Teraz zamieniam docs na corpus, �eby u�y� funkcji z pakietu tm do policzenia tf-idf
# Pr�bowa�em wcze�niej liczy� to r�cznie, ale si� liczy�o kilka godzin i si� nie doliczy�o
# a w pakiecie tm si� natychmiast
myCorpus = Corpus(DataframeSource(docs),
                   readerControl = list(reader = readDataframe, language = "en"))

DTM = DocumentTermMatrix(myCorpus, 
                          control = list(weighting = weightTfIdf))
# Jest tylko jeden problem - DTM zawiera mniej s��w ni� nasz word_vectors_skipgram.
# Trzeba znale�� te dodatkowe s�owa i usun�� z docs

wordsDTM = colnames(DTM) #To s� wszystkie s�owa wyst�puj�ce w DTM
wordsSkipgram = rownames(word_vectors_skipgram) #To s� wszystkie s�owa wyst�puj�ce w word_vectors_skipgram

length(wordsDTM)
length(wordsSkipgram)

# S�owa kt�re s� w Skipgram, a nie ma ich w DTM
missing_words = lapply(wordsSkipgram,function(x, words) if(!(x%in%words)){return(x)},wordsDTM)
missing_words = missing_words[-which(sapply(missing_words, is.null))]
# W missing_words s� g��wnie dwuliterowe s�owa, oraz kilka pojedy�czych liter

#S�owa kt�re s� w DTM, a nie ma ich w Skipgram
missing_words2 = lapply(wordsDTM,function(x, words) if(!(x%in%words)){return(x)},wordsSkipgram)
missing_words2 = missing_words2[-which(sapply(missing_words2, is.null))]
# S� tu prawie 93 tysi�ce s��w. Dlaczego nie ma ich w word_vectors_skipgram? Nie mam poj�cia.
# By� mo�e dlatego, �e wyst�powa�y za rzadko.
# Natomiast "nadmiarowe" missing_words2 nie s� problemem (nie musz� si� do nich odwo�ywa�)
# Problemem s� missing_words, kt�re musz� usun�� z docs

tmp = sapply(docs[,text], function(x, missing) removeWords(unlist(x), missing), missing_words)
#Uwaga: powy�sze na razie nie dzia�a.


###Dalsza idea by�a taka, �eby wzi�� ka�dy dokument zakodowa� w spos�b:
#  - ka�de s�owo zakodowa� odpowiadaj�cym mu wektorem z word_vectors_skipgram
#  - ka�dy z tych wektor�w pomno�y� przez tf-idf tego s�owa w tym dokumencie.
#  - zsumowa�
# Wygl�da�oby to mniej wi�cej tak, ale nie wiem, czy dzia�a, bo nie dotar�em do tego momentu
# Wcze�niej, poniewa� skipgram i DTM mia�y inne zbiory s��w, to wywala�a si� linijka inspect(DTM[i,row.names(wordVec)])
doc_embeddings = matrix( rep( 0, len=100*100000), nrow = 100000)
for(i in 1:nrows(DTM)){
  row = strsplit(docs[i, text], " ")[[1]]
  wordVec = word_vectors_skipgram[[row, average=F]] #wordVec to macierz 100 na (liczba s��w w dokumencie)
  wordVec = wordVec * inspect(DTM[i,row.names(wordVec)])
  doc_embeddings[i,] = apply(wordVec, 2, sum)
}

