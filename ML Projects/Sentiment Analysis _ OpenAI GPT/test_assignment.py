from openai import OpenAI
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

DEEPGRAM_API_KEY = 'API KEY'
api_key = 'API KEY'
client = OpenAI(api_key=api_key)

def analyze_sentiment(text):
    # prompt = "The following is a sentiment analysis of the given text:\n\n" + text + "\n\nSentiment:"
    print("Input Text: ", text)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "system",
            "content": "You will be provided with a series of conversation to analyze the sentiment and psychological behaviour."
                       "I would like you to provide an brief psychological insights about each speaker in few words"

          },
          {
            "role": "user",
            "content": text
          }
        ],
        # prompt=prompt,
        max_tokens=50,
        temperature=0.8,
        top_p=1,
        stop="\n"
    )
    return response.choices[0].message.content


def speechtotext(src):

    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    with open(src, 'rb') as audio:
        buffer_data = audio.read()
    # print(buffer_data)

    payload: FileSource = {"buffer": buffer_data}
    options = PrerecordedOptions(model="nova-2", smart_format=True, utterances=True, punctuate=True, diarize=True)
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload,options=options)

    # print(response)
    resp_transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]   # Extracts transcript from response
    resp_paragraph = response["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"]  # Extracts paragraph transcript from response
    f = open('transcribed_audio.txt','w')
    f.write(resp_paragraph)
    f.close()
    para_li = ''.join(resp_paragraph).split('\n')
    # print(para_li)
    resp_paragraph_li = [line.split("''") for line in para_li]
    return resp_transcript, resp_paragraph_li

if __name__ == "__main__":
    # text_to_analyze = input("Enter the text to analyze sentiment: ")
    prompt = "[Speaker_1]: Iam good, and gonna hit the gym. how about you?"
    sentiment = analyze_sentiment(prompt)
    print("Sentiment:", sentiment)
    transcript, para = speechtotext("CallCenterPhoneCall.mp3")
    print(para)
