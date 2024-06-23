from dash import html, dcc, dash_table, Output, Input, State
import dash
from dash_recording_components import AudioRecorder
import soundfile as sf
import dash_bootstrap_components as dbc
import io
import base64
import numpy as np
import pandas as pd
import wave
import dash_auth
from audio2vec import Audio2Vec

Annotated = pd.read_csv("Annotated.csv")

audioProcessor = Audio2Vec()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

VALID_USERNAME_PASSWORD_PAIRS = {
    'test': 'test123'
}

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Row([
                html.Label(
                    " Audio Annotation Tool",
                    style={
                        "font-family": "Times Roman",
                        "text-decoration": "underline",
                        "text-align": "center",
                        "color": "black",
                        "margin": "20px",
                    },
                ),
            ]),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Br(),
                        dbc.CardImg(
                            src="/assets/logoA.png",
                            top=True,
                            style={
                                "width": "auto",
                                "height": "auto",
                                "display": "block",
                                "margin-left": "auto",
                                "margin-right": "auto",
                            },
                        ),
                    ]
                ),
                style={
                    "width": "21.6rem",
                    "margin": "0 auto",
                    "border": "2px solid gold",
                    "background-color": "#f8f9fa",
                },
            )],
            style={"margin": "20px", "text-align": "center"},
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Button("Record", id="record-button", className="btn btn-primary", style={"margin": "10px"}),
                dbc.Button(
                    "Stop Recording",
                    id="stop-button",
                    n_clicks=0,
                    className="btn btn-secondary",
                    style={"margin": "10px"},
                ),
                dbc.Button("Play", id="play-button", className="btn btn-success", style={"margin": "10px"}),
            ],
            style={"margin": "20px", "text-align": "center", "padding": "10px"},
        ),
        html.Div(
            id="audio-output",
            style={"margin": "10px", "text-align": "center"},
        ),
        html.Div(
            id="dummy-output",
            style={"display": "none", "margin": "10px", "border": "1px solid gold", "background-color": "#f8f9fa"},
        ),
        AudioRecorder(id="audio-recorder"),
        html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i, "renamable": True, "hideable": True} for i in Annotated.columns],
                data=Annotated.to_dict('records'),
                style_table={'overflowX': 'auto'},
                export_format='xlsx',
                editable=True,
                include_headers_on_copy_paste=True,
                sort_action='native',
                page_action="native",
                page_size=5,
                style_cell={
                    'height': 'auto',
                    'minWidth': '140px', 'width': '150px', 'maxWidth': '180px',
                    'whiteSpace': 'normal'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'color': 'black'
                },
                style_data={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'color': 'black'
                }
            ),
            dcc.Store(id='table-data-store', data=Annotated.to_dict('records')),
        ])
    ],
    style={
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "center",
        "align-items": "center",
        "height": "100vh",
    },
    className="container-fluid",
)

audio_samples = []
server_data = []


@app.callback(
    Output('table-data-store', 'data'),
    Input('table', 'data'),
    prevent_initial_call=True
)
def update_store_data(rows):
    updated_df = pd.DataFrame(rows)
    updated_df.to_csv("Annotated.csv", index=False)
    return rows


@app.callback(
    Output('table', 'data'),
    Input('table-data-store', 'modified_timestamp'),
    State('table-data-store', 'data'),
    prevent_initial_call=True
)
def update_table_data(ts, data):
    df = pd.read_csv("Annotated.csv")
    return df.to_dict('records')


@app.callback(
    Output("audio-recorder", "recording"),
    Input("record-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State("audio-recorder", "recording"),
    prevent_initial_call=True
)
def control_recording(record_clicks, stop_clicks, recording):
    return record_clicks > stop_clicks


@app.callback(
    Output("audio-output", "children"),
    Input("play-button", "n_clicks"),
    prevent_initial_call=False
)
def play_audio(play_clicks):
    if play_clicks:
        if audio_samples:
            audio_array = np.array(audio_samples)
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio_array, 16000, format="WAV")
                wav_bytes = wav_buffer.getvalue()
                wav_base64 = base64.b64encode(wav_bytes).decode()
                audio_src = f"data:audio/wav;base64,{wav_base64}"
                wavFile = "./img/outofsample-077.wav"
                bytearray_data = bytearray(audio_array)
                with wave.open(wavFile, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(bytearray_data)
                dF0 = audioProcessor.audio2Vec2DfProcessor(wavFile)
                detectDf = pd.DataFrame([], columns=['Prediction'])
                dF0 = pd.concat([dF0, detectDf], ignore_index=True, axis=1)
                new_column_names = {
                    0: 'features-1',
                    1: 'features-2',
                    2: 'features-3',
                    3: 'features-4',
                    4: 'features-5',
                    5: 'features-6',
                    6: 'features-7',
                    7: 'features-8',
                    8: 'features-9',
                    9: 'features-10',
                    10: 'features-11',
                    11: 'prediction'
                }
                dF0.rename(columns=new_column_names, inplace=True)
                dF0.to_csv("Annotated.csv", index=True, mode="a", header=False)

                return html.Audio(src=audio_src, controls=True, autoPlay=True, style={
                    "margin": "20px"
                }),
    return dash.no_update


@app.callback(
    Output("dummy-output", "children"),
    Input("audio-recorder", "audio"),
    prevent_initial_call=True
)
def update_audio(audio):
    global audio_samples
    if audio is not None:
        audio_samples += list(audio.values())
    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True, port=5979)
