import openai
from openai import OpenAI
api_key = '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'
client = OpenAI(api_key=api_key)

def analyze_sentiment(text):
    # prompt = "The following is a sentiment analysis of the given text:\n\n" + text + "\n\nSentiment:"
    print("Text: ", text)
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
        temperature=0.7,
        top_p=1,
        stop="\n"
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # text_to_analyze = input("Enter the text to analyze sentiment: ")
    prompt = "[Speaker_1]: Iam good, and gonna hit the gym. how about you?"
    sentiment = analyze_sentiment(prompt)
    print("Sentiment:", sentiment)
