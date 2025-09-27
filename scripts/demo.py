import gradio as gr
from main import analyze_sentence 

def predict(sentence):
    return analyze_sentence(sentence)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence..."),
    outputs="json",
    title="Aspect-Based Sentiment Analysis",
    description="Extracts aspects and classifies sentiment (positive, negative, neutral)."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
