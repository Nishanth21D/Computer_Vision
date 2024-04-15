import numpy as np
from flask import Flask, render_template, request
import easygui
from test_assignment import analyze_sentiment, speechtotext
app = Flask(__name__)

def get_attachment_name(file):
    mp3_file = False
    if file.filename != '':
        filename = str(file.filename)
        print(filename, type(filename))
        ext = filename[-3::]
        if ext in ['txt', 'mp3', 'wav']:
            if ext in ['mp3', 'wav']:
                print('__mp3__')
                mp3_file = True
                return filename, mp3_file
            return filename, mp3_file
    else:
        return None, mp3_file

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'attachment' in request.files:
            # attachment = request.files['attachment']
            attachment = request.files.get("attachment")
            uploaded_file, mp3_file = get_attachment_name(attachment)

            if uploaded_file:
                if mp3_file:
                    # print("_____In____")
                    transcript, file_content = speechtotext(uploaded_file)
                else:
                    file_content = [line.decode('utf-8').splitlines(keepends=False) for line in attachment]
                # print('content',file_content)
                query_list = []
                sent_list = []
                for i,line in enumerate(file_content):
                    if len(line[0]) == 0:               # ignoring empty space
                        continue
                    # print(line[0])
                    sent_list.append(analyze_sentiment(line[0]))
                    query_list.append(line[0])
                    # if i != len(file_content):
                    #     easygui.msgbox("Prompt Running, Please Wait !!","Loading",)
                    # easygui.textbox("Prompt Running, Please Wait !!","Loading",sent_list[i])
                    # easygui.exceptionbox("Prompt Running, Please Wait !!","Loading")
                # print(sent_list)
                write_sent = " ".join(sent_list)
                sent_list = np.transpose(np.array(sent_list))
                query_list = np.transpose(np.array(query_list))
                attachment_content = query_list

                # Writing the output in the text file
                fil_name = str(f"{uploaded_file[:-4:]}_output.txt")
                print(fil_name)
                f = open(fil_name, 'w')
                f.write(write_sent)
                f.close()

            else:
                attachment_content = None
        else:
            uploaded_file = None
            attachment_content = None

        # Show the output in a dialog box
        output_message = (f'Uploaded file: {uploaded_file}\n\nAttachment Content:\n{attachment_content}\n\n'
                          f'Sentiment Analysis:\n{sent_list}')
        easygui.msgbox(output_message, title="Output")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
