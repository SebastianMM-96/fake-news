import pickle

news = input('Enter a news text to classify: \n')

def classFakeNews(news):
    # Using the model
    loadModel = pickle.load(open('<model_name>', 'rb'))
    prediction = loadModel.predict([news])

    return(print('The news is: {}'.format(prediction[0])))

if __name__ == "__main__":
    classFakeNews(news)