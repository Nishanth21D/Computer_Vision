import numpy as np
from flask import Flask, render_template, request
import easygui
from test_assignment import analyze_sentiment
app = Flask(__name__)

def get_attachment_name(file):
    if file.filename != '':
        return file.filename
    else:
        return None
def read_attachment_content(file):
    if file:
        return file.read()
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # Check if the post request has the file part
        if 'attachment' in request.files:
            # attachment = request.files['attachment']
            attachment = request.files.get("attachment")
            uploaded_file = get_attachment_name(attachment)
            if uploaded_file:
                file_content = [line.decode('utf-8').splitlines(keepends=False) for line in attachment]
                # print('content',file_content)
                query_list = []
                sent_list = []
                for i,line in enumerate(file_content):
                    sent_list.append(analyze_sentiment(line[0]))
                    query_list.append(file_content[i][0])
                    # if i != len(file_content):
                    #     easygui.msgbox("Prompt Running, Please Wait !!","Loading",)
                        # easygui.textbox("Prompt Running, Please Wait !!","Loading",sent_list[i])
                        # easygui.exceptionbox("Prompt Running, Please Wait !!","Loading")
                sent_list = np.transpose(np.array(sent_list))
                print(sent_list)
                query_list = np.transpose(np.array(query_list))
                attachment_content = query_list

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
