%%shell
set -e

#---------------------------------------------------#
JULIA_VERSION="1.5.3" # any version ≥ 0.7.0
JULIA_PACKAGES="IJulia BenchmarkTools Plots"
JULIA_PACKAGES_IF_GPU="CuArrays"
JULIA_NUM_THREADS=2
#---------------------------------------------------#

if [ -n "$COLAB_GPU" ] && [ -z `which julia` ]; then
  # Install Julia
  JULIA_VER=`cut -d '.' -f -2 <<< "$JULIA_VERSION"`
  echo "Installing Julia $JULIA_VERSION on the current Colab Runtime..."
  BASE_URL="https://julialang-s3.julialang.org/bin/linux/x64"
  URL="$BASE_URL/$JULIA_VER/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
  wget -nv $URL -O /tmp/julia.tar.gz # -nv means "not verbose"
  tar -x -f /tmp/julia.tar.gz -C /usr/local --strip-components 1
  rm /tmp/julia.tar.gz

  # Install Packages
  if [ "$COLAB_GPU" = "1" ]; then
      JULIA_PACKAGES="$JULIA_PACKAGES $JULIA_PACKAGES_IF_GPU"
  fi
  for PKG in `echo $JULIA_PACKAGES`; do
    echo "Installing Julia package $PKG..."
    julia -e 'using Pkg; pkg"add '$PKG'; precompile;"'
  done

  # Install kernel and rename it to "julia"
  echo "Installing IJulia kernel..."
  julia -e 'using IJulia; IJulia.installkernel("julia", env=Dict(
      "JULIA_NUM_THREADS"=>"'"$JULIA_NUM_THREADS"'"))'
  KERNEL_DIR=`julia -e "using IJulia; print(IJulia.kerneldir())"`
  KERNEL_NAME=`ls -d "$KERNEL_DIR"/julia*`
  mv -f $KERNEL_NAME "$KERNEL_DIR"/julia  

  echo ''
  echo "Success! Please reload this page and jump to the next section."
fi

versioninfo()

import Pkg

Pkg.add("Word2Vec")
Pkg.add("Gadfly")
Pkg.add("TextAnalysis")
Pkg.add("Distances")
Pkg.add("Statistics")
Pkg.add("MultivariateStats")
Pkg.add("PyPlot")
Pkg.add("WordTokenizers")
Pkg.add("DelimitedFiles")
Pkg.add("Plots")

using Word2Vec
using Distances, Statistics
using MultivariateStats
using PyPlot
using Gadfly
using WordTokenizers
using TextAnalysis
using DelimitedFiles

function load_embeddings(embedding_file)
    local LL, indexed_words, index
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line)
            word = xs[1]
            push!(indexed_words, word)
            push!(LL, parse.(Float32, xs[2:end]))
            index += 1
        end
    end
    return reduce(hcat, LL), indexed_words
end

function vec(s) 
    if glove_vec_idx(s) != nothing
        embeddings[:, glove_vec_idx(s)]
    end    
end

function closest(v, n=11)
    list=[(x,cosine(embeddings'[x,:], v)) for x in 1:size(embeddings)[2]]
    topn_idx = sort(list, by = x -> x[2], rev = true)[1:n]
    return [vocab[a] for (a,_) in topn_idx]
end

function sentvec(s) 
    local arr=[]
    for w in split(sentences[s])
        if vec(w)!=nothing
            push!(arr, vec(w))
        end
    end
    if length(arr)==0
        ones(Float32, (50,1))*999
    else
        mean(arr)
    end
end

function closest_sent(input_str, n=20)
    mean_vec_input=mean([vec(w) for w in split(input_str)])
    list=[(x,cosine(mean_vec_input, sentvec(x))) for x in 1:length(sentences)]
    topn_idx=sort(list, by = x -> x[2], rev=true)[1:n]
    return [sentences[a] for (a,_) in topn_idx]
end

function closest_sent_pretrained(pretrained_arr, input_str, n=20)
    mean_vec_input=mean([vec(w) for w in split(input_str)])
    list=[(x,cosine(mean_vec_input, pretrained_arr[x,:])) for x in 1:length(sentences)]
    topn_idx=sort(list, by = x -> x[2], rev=true)[1:n]
    return [sentences[a] for (a,_) in topn_idx]
end

# create a vectorized txt file
word2vec("text8", "text8-vector.txt", verbose=true)

# create a model for text8
text8_model = wordvectors("text8-vector.txt")

# describe the model
text8_vec_size, text8_vocab_size = size(text8_model)
println("Loaded embeddings for text8-vector file, each word is represented by a vector with $text8_vec_size features. The vocab size is $text8_vocab_size")

# create a vectorized txt file that houses the phrases
word2phrase("text8", "text8phrase")
word2vec("text8phrase", "text8phrase-vector.txt", verbose=true)

# create a model based on text8phrase data
text8_model_phrases = wordvectors("text8phrase-vector.txt")

# describe the model based on text8phrase data
text8_phrases_vec_size, text8_phrases_vocab_size = size(text8_model_phrases)
println("Loaded embeddings for text8 phrase-vector file, each word is represented by a vector with $text8_phrases_vec_size features. The vocab size is $text8_phrases_vocab_size")

# create an indexed cluster file using the text8 data
word2clusters("text8", "text8-cluster.txt", 100)

text8_model_cluster = wordclusters("text8-cluster.txt")

# describe the model
text8_clusters_vec_size, text8_clusters_vocab_size = clusters(text8_model_cluster)
println("Loaded embeddings for text8 clusters file, each word is represented by a vector with $text8_clusters_vec_size features. The vocab size is $text8_clusters_vocab_size")

embeddings, vocab = load_embeddings("glove.6B.50d.txt")
glove_vec_size, glove_vocab_size = size(embeddings)
println("Loaded embeddings for Glove.6B.50d file, each word is represented by a vector with $glove_vec_size features. The vocab size is $glove_vocab_size")

# check index number of given word, in this case "cheese"
glove_vec_idx(s) = findfirst(x -> x==s, vocab)
println("The index number for 'cheese' is: ", glove_vec_idx("cheese"))

text8_words = vocabulary(text8_model)

# check index number of given word, in this case "hungary"
test8_vec_idx(s) = findfirst(x -> x==s, text8_words)
println("The index number for 'hungary' is: ", test8_vec_idx("hungary"))

vec("cheese")

vec("queen")

similarity(text8_model, "jupiter", "neptune")

similarity(text8_model, "santa", "snow")


similarity(text8_model, "hungary", "neptune")


similarity(text8_model, "milk", "cheese")

similarity(text8_model, "nutmeg", "football")

similarity(text8_model, "book", "sugar")

similarity(text8_model, "maroon", "messages")

Word2Vec.cosine(x,y)=1-cosine_dist(x, y)

println("is the similarity between 'dog' and 'cat' greater than the similarity between 'moon' and 'man'?\n", 
"----> ", cosine(vec("dog"), vec("cat")) > cosine(vec("moon"), vec("man")))

# find similar words to 'hungary', limit to 22 words
text8_vec_idx_1, text8_dists = cosine(text8_model, "hungary", 22)

Gadfly.plot(x = text8_words[text8_vec_idx_1], y = text8_dists)

closest(vec("wine"))

closest(vec("water") + vec("frozen"))

closest(mean([vec("day"), vec("night")]))

blue_to_sky = vec("blue") - vec("sky")
closest(blue_to_sky + vec("grass"))

closest(vec("man") - vec("woman") + vec("queen"))


closest(vec("king") - vec("queen") + vec("woman"))


analogy_words(text8_model, ["hungary", "estonia"], ["belarus"], 10)

analogy_words(text8_model, ["tree", "leaf"], ["bark"], 10)

cosine_similar_words(text8_model_phrases, "new_jersey", 5)

txt = open("macbeth.txt") do file
    read(file, String)
end
println("Loaded Macbeth, length = $(length(txt)) characters")

txt = replace(txt, r"\n|\r|_|," => " ")
txt = replace(txt, r"[\"*();!]" => "")
sd=StringDocument(txt)
prepare!(sd, strip_whitespace)
sentences = split_sentences(sd.text)
i=1
for s in 1:length(sentences)
    if length(split(sentences[s]))>3
        sentences[i]=lowercase(replace(sentences[s], "."=>""))
        i+=1
    end
end
sentences[1000:1010]

sentences[22]

sentvec(22)

closest_sent("my favorite food is strawberry ice cream")

macbeth_sent_vecs=[]
for s in 1:length(sentences)
    i==1 ? macbeth_sent_vecs=sentvec(s) : push!(macbeth_sent_vecs,sentvec(s))
end

writedlm( "macbeth_sent_vec.csv",  macbeth_sent_vecs, ',')


writedlm( "macbeth_sentences.csv",  sentences, ',')


sentences=readdlm("macbeth_sentences.csv", '!', String, header=false)
macbeth_sent_vecs=readdlm("macbeth_sent_vec.csv", ',', Float32, header=false)

closest_sent_pretrained(macbeth_sent_vecs, "i stabbed him in the back")
