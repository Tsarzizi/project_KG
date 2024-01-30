import gradio as gr
from test import predict
from searcher import AnswerSearcher

searcher = AnswerSearcher()
def inference(question):
    output = predict(question)
    disease = output[0]
    question_type = output[1]
    answer = searcher.get_answer(question_type, disease)
    if not answer:
        answer = ('本系统未能理解您所提出的问题或者是数据库内并未此类问题答案，请更换提问方式！')
    return answer

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">医疗知识图谱问答系统</h1>""")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...").style(container=False)
                output = gr.Textbox(show_label=False, lines=5).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")

    # history = gr.State([])
    submitBtn.click(inference, user_input, output, show_progress=True)
    # submitBtn.click(reset_user_input, [], [user_input])

    # emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
'''
with gr.Blocks() as demo:
    name = gr.Textbox()
    output = gr.Textbox()
    greet_btn = gr.Button('Submit')
    greet_btn.click(fn=inference, inputs=name, outputs=output)
'''
demo.launch(share=False, inbrowser=True)