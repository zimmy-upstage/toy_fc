import gradio as gr

# Define a function to change the active tab
def change_tab():
    # Return the updated Tabs component with the selected tab index
    return gr.Tabs.update(selected=1)  # Change to the tab with index 1

# Create a Gradio Blocks interface
with gr.Blocks() as demo:
    # Define Tabs component
    with gr.Tabs() as tabs:
        # Define first tab with an ID of 0
        with gr.TabItem("Train", id=0):
            t = gr.Textbox(label="Training Textbox")
        # Define second tab with an ID of 1
        with gr.TabItem("Inference", id=1):
            i = gr.Image(label="Inference Image")
    
    # Add a button that will trigger the change_tab function on click
    btn = gr.Button("Switch Tab")
    btn.click(change_tab, None, tabs)

# Launch the Gradio interface
demo.launch()
