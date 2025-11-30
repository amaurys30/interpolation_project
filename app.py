# app.py (final)
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io, base64
import warnings
warnings.filterwarnings("ignore")

# ----------------- MÉTRICAS -----------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2))) if len(y_true)>0 else float('nan')
def mae(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true)>0 else float('nan')
def max_error(y_true, y_pred):
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred)))) if len(y_true)>0 else float('nan')
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    if len(y_true)==0: return float('nan')
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot) if ss_tot != 0 else float('nan')

# ----------------- INTERPOLADORES / UTIL -----------------
def poly_predict_from_coeffs(coeffs, xq):
    p = np.poly1d(coeffs)
    return p(xq)

def lagrange_evaluator_degree(xs, ys, degree):
    xs = np.asarray(xs); ys = np.asarray(ys)
    n = len(xs)
    degree = min(degree, n-1)
    def eval_fn(xq):
        xq = np.atleast_1d(xq).astype(float)
        out = np.zeros_like(xq, dtype=float)
        for k, xval in enumerate(xq):
            idx = np.argsort(np.abs(xs - xval))[:degree+1]
            xi = xs[idx]; yi = ys[idx]
            val = 0.0
            for i in range(len(xi)):
                Li = 1.0
                for j in range(len(xi)):
                    if i==j: continue
                    Li *= (xval - xi[j])/(xi[i] - xi[j])
                val += yi[i]*Li
            out[k] = val
        return out
    return eval_fn

def newton_divided_coeffs(xs, ys, degree):
    xs = np.array(xs); ys = np.array(ys)
    n = len(xs)
    dd = np.zeros((n, n))
    dd[:,0] = ys
    for j in range(1,n):
        for i in range(n-j):
            dd[i,j] = (dd[i+1,j-1] - dd[i,j-1]) / (xs[i+j] - xs[i])
    coeffs = dd[0,:]
    degree = min(degree, n-1)
    selected = coeffs[:degree+1]
    def evaluator(xq):
        xq = np.atleast_1d(xq).astype(float)
        out = np.full_like(xq, selected[0], dtype=float)
        for i in range(1, len(selected)):
            term = selected[i] * np.ones_like(xq, dtype=float)
            for j in range(i):
                term *= (xq - xs[j])
            out += term
        return out
    return evaluator

def linear_piecewise(xs, ys):
    def f(xq):
        return np.interp(xq, xs, ys)
    return f

def inverse_interpolation_polynomial(xs, ys, degree, y_targets):
    coeffs = np.polyfit(xs, ys, degree)
    xs_out = []
    for yt in np.atleast_1d(y_targets):
        coeffs_mod = coeffs.copy()
        coeffs_mod[-1] -= yt
        roots = np.roots(coeffs_mod)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
        if len(real_roots)>0:
            chosen = min(real_roots, key=lambda r: abs(r - np.mean(xs)))
            xs_out.append(chosen)
        else:
            xs_out.append(np.nan)
    return np.array(xs_out), coeffs

# ----------------- PARSE CSV -----------------
def parse_contents(contents, filename):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.BytesIO(decoded))
        if 'x' in df.columns and 'y' in df.columns:
            return df[['x','y']]
        else:
            return df.iloc[:,0:2].rename(columns={df.columns[0]:'x', df.columns[1]:'y'})
    except Exception:
        return None

# ----------------- APP & LAYOUT -----------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5', 'padding': '10px'}, children=[

    # HEADER
    html.Div(style={
        'backgroundColor': '#2c3e50',
        'padding': '12px',
        'borderRadius': '8px',
        'marginBottom': '12px',
        'color': 'white',
        'textAlign': 'center'
    }, children=[
        html.H1("Interpolación y Extrapolación ", style={'margin': '0', 'fontSize':'24px'}),
        html.Div("Proyecto Final · Análisis de Técnicas Numéricas — 2025")
    ]),

    # BODY (flex)
    html.Div(style={'display': 'flex', 'gap': '12px'}, children=[

        # LEFT PANEL (Inputs) - SCROLL INDEPENDIENTE
        html.Div(style={
            'width': '25%', 'backgroundColor': 'white', 'padding': '12px', 'borderRadius': '8px',
            'boxShadow': '0px 0px 6px rgba(0,0,0,0.12)', 'overflowY': 'auto', 'maxHeight': '780px'
        }, children=[
            html.H3("Entrada de datos", style={'marginTop':0}),

            html.Br(),
            html.Button("Acerca de", id="open-about", style={
                'width':'100%','padding':'10px','background':'#34495e','color':'white','border':'none','borderRadius':'5px'
            }),
            html.Div(id="about-modal"),

            dcc.Upload(id='upload-data', children=html.Button('Cargar archivo CSV', style={
                'width':'100%','padding':'8px','background':'#3498db','color':'white','border':'none','borderRadius':'5px'
            })),
            html.Div("CSV: columnas 'x' y 'y' (o las dos primeras columnas serán usadas)"),
            html.Hr(),
            html.H4("Puntos a evaluar"),
            html.Label("Valores de x (separados por coma) — opcional:"),
            dcc.Input(id='xq-input', type='text', placeholder='Ej: 1.2,2.5,5.0', style={'width':'100%','padding':'6px'}),
            html.Br(), html.Br(),
            html.H4("Métodos (vistos en clase)"),
            dcc.Checklist(
                id='methods-checklist',
                options=[
                    {'label': 'Lineal', 'value': 'linear'},
                    {'label': 'Cuadrática', 'value': 'quad'},
                    {'label': 'Cúbica', 'value': 'cubic'},
                    {'label': 'Lagrange deg1', 'value': 'lag1'},
                    {'label': 'Lagrange deg2', 'value': 'lag2'},
                    {'label': 'Lagrange deg3', 'value': 'lag3'},
                    {'label': 'Newton deg1', 'value': 'new1'},
                    {'label': 'Newton deg2', 'value': 'new2'},
                    {'label': 'Newton deg3', 'value': 'new3'},
                    {'label': 'Interpolación inversa (pol deg3)', 'value': 'inv3'}
                ],
                value=['linear','lag1','lag2','lag3','new1','new2','new3']
            ),
            html.Br(),
            html.Label("Modo de graficado:"),
            dcc.RadioItems(id='plot-mode', options=[
                {'label':'Todas (overlay + individuales)','value':'all'},
                {'label':'Sólo overlay','value':'overlay'},
                {'label':'Sólo individuales','value':'individuals'}
            ], value='all'),
            html.Br(),
            html.Button("Ejecutar métodos", id='run-button', style={
                'width':'100%','padding':'10px','background':'#27ae60','color':'white','border':'none','borderRadius':'5px'
            }),
            html.Br(), html.Br(),
            html.H4("Interpolación inversa"),
            html.Label("Valores de y (separados por coma) — opcional:"),
            dcc.Input(id='y_targets', type='text', placeholder='Ej: 2.5,4.0', style={'width':'100%','padding':'6px'}),
            html.Br(), html.Br(),
            html.Br(), html.Br(),
            html.Div("Nota: Las tablas individuales muestran los pares (x_evaluado, y_estimado) para cada método.")
        ]),

        # CENTER PANEL (Graphs) - SCROLL INDEPENDIENTE
        html.Div(style={
            'width': '50%', 'backgroundColor': 'white', 'padding': '12px', 'borderRadius': '8px',
            'boxShadow': '0px 0px 6px rgba(0,0,0,0.12)', 'overflowY': 'auto', 'maxHeight': '780px'
        }, children=[
            html.H3("Gráficas", style={'marginTop':0}),
            dcc.Graph(id='main-graph', style={'height':'380px'}),
            html.H4("Gráficas individuales (cada método tiene su tabla debajo)"),
            html.Div(id='individual-graphs', style={'marginTop':'8px'})
        ]),

        # RIGHT PANEL (Results + combined values) - SCROLL INDEPENDIENTE
        html.Div(style={
            'width': '25%', 'backgroundColor': 'white', 'padding': '12px', 'borderRadius': '8px',
            'boxShadow': '0px 0px 6px rgba(0,0,0,0.12)', 'overflowY': 'auto', 'maxHeight': '780px'
        }, children=[
            html.H3("Resultados (Errores)", style={'marginTop':0}),
            dash_table.DataTable(
                id='results-table',
                columns=[
                    {"name": "Método", "id": "method"},
                    {"name": "RMSE", "id": "rmse"},
                    {"name": "MAE", "id": "mae"},
                    {"name": "MaxErr", "id": "maxerr"},
                    {"name": "R2", "id": "r2"},
                    {"name": "Nota", "id": "note"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '6px','fontSize':'12px'},
                style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold'},
                page_size=8
            ),
            html.Br(),
            html.H4("Tabla combinada de valores (todos los métodos)"),
            html.Div(id='combined-table-container')
        ])
    ]),

    # store para guardar último run (para exportar)
    dcc.Store(id='last-run-store'),
    dcc.Store(id='stored-data'),
    #dcc.Download(id="download-results")

])

# ----------------- CALLBACK: almacenar CSV cargado -----------------
@app.callback(Output('stored-data','data'),
              Input('upload-data','contents'),
              State('upload-data','filename'))
def store_upload(contents, filename):
    if contents is None:
        # dataset demo por defecto
        xs = np.linspace(0,5,6)
        ys = np.array([1,2.5,3.2,5.0,7.5,11.0])
        df = pd.DataFrame({'x':xs,'y':ys})
        return df.to_dict('records')
    df = parse_contents(contents, filename)
    if df is None:
        return None
    return df.to_dict('records')

# ----------------- CALLBACK PRINCIPAL -----------------
@app.callback(
    Output('main-graph','figure'),
    Output('results-table','data'),
    Output('individual-graphs','children'),
    Output('combined-table-container','children'),
    Output('last-run-store','data'),
    Input('run-button','n_clicks'),
    State('stored-data','data'),
    State('methods-checklist','value'),
    State('xq-input','value'),
    State('plot-mode','value'),
    State('y_targets','value'),
    prevent_initial_call=True
)
def run_methods(n_clicks, records, methods, xq_text, plot_mode, y_targets):
    if records is None:
        return dash.no_update
    df = pd.DataFrame.from_records(records)
    xs = df['x'].values.astype(float)
    ys = df['y'].values.astype(float)
    x_plot = np.linspace(xs.min(), xs.max(), 600)

    # parse xq
    xq = []
    if xq_text:
        try:
            xq = [float(s) for s in xq_text.split(',') if s.strip()!='']
        except:
            xq = []

    # parse y targets
    yts = []
    if y_targets:
        try:
            yts = [float(s) for s in y_targets.split(',') if s.strip()!='']
        except:
            yts = []

    overlay_traces = [go.Scatter(x=xs, y=ys, mode='markers', name='Datos', marker={'size':7,'color':'black'})]
    results = []
    individual_children = []
    combined_rows = []

    # helper: create values table df
    def create_values_table_df(x_vals, y_vals):
        dfv = pd.DataFrame({'x_evaluado': np.round(x_vals, 8), 'y_estimado': np.round(y_vals, 8)})
        return dfv

    def add_individual_block(method_name, xvals_for_plot, yvals_for_plot, xvals_table, yvals_table):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xvals_for_plot, y=yvals_for_plot, mode='lines', name=method_name))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name='Datos', marker={'size':6,'color':'black'}))
        if len(xq)>0:
            fig.add_trace(go.Scatter(x=xvals_table, y=yvals_table, mode='markers', name='Puntos evaluados', marker={'size':9,'symbol':'x'}))
        fig.update_layout(title=method_name, margin={'t':30, 'b':5}, height=300)
        df_table = create_values_table_df(xvals_table, yvals_table)
        datatable = dash_table.DataTable(
            columns=[{"name":"x_evaluado","id":"x_evaluado"}, {"name":"y_estimado","id":"y_estimado"}],
            data=df_table.to_dict('records'),
            style_cell={'textAlign':'center','padding':'4px','fontSize':'12px'},
            style_header={'backgroundColor':'#f1f1f1','fontWeight':'bold'},
            style_table={'overflowX':'auto'},
            page_size=6
        )
        block = html.Div(style={'marginBottom':'12px','paddingBottom':'8px','borderBottom':'1px solid #eee'}, children=[
            dcc.Graph(figure=fig, style={'height':'300px'}),
            html.Div(datatable)
        ])
        return block, df_table

    # loop over methods
    for m in methods:
        note = ''
        if m == 'linear':
            f = linear_piecewise(xs, ys)
            y_plot = f(x_plot)
            y_at_xs = f(xs)
            overlay_traces.append(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Lineal'))
            x_table = np.array(xq) if xq else xs
            y_table = f(x_table)
            results.append({'method':'Lineal','rmse':round(rmse(ys,y_at_xs),8),'mae':round(mae(ys,y_at_xs),8),'maxerr':round(max_error(ys,y_at_xs),8),'r2':round(r2_score(ys,y_at_xs),8),'note':note})
            if plot_mode in ('all','individuals'):
                block, df_tab = add_individual_block('Lineal', x_plot, y_plot, x_table, y_table)
                individual_children.append(block)
            for xx, yy in zip(x_table, y_table):
                combined_rows.append({'method':'Lineal','x_evaluado':float(xx),'y_estimado':float(yy)})

        if m == 'quad':
            deg = 2
            if len(xs) < 3:
                note = 'Pocos datos para grado 2; se usa grado disponible.'
                deg = min(2, len(xs)-1)
            coeffs = np.polyfit(xs, ys, deg)
            y_plot = poly_predict_from_coeffs(coeffs, x_plot)
            y_at_xs = poly_predict_from_coeffs(coeffs, xs)
            overlay_traces.append(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Cuadrática'))
            x_table = np.array(xq) if xq else xs
            y_table = poly_predict_from_coeffs(coeffs, x_table)
            results.append({'method':'Cuadrática','rmse':round(rmse(ys,y_at_xs),8),'mae':round(mae(ys,y_at_xs),8),'maxerr':round(max_error(ys,y_at_xs),8),'r2':round(r2_score(ys,y_at_xs),8),'note':note})
            if plot_mode in ('all','individuals'):
                block, df_tab = add_individual_block('Cuadrática', x_plot, y_plot, x_table, y_table)
                individual_children.append(block)
            for xx, yy in zip(x_table, y_table):
                combined_rows.append({'method':'Cuadrática','x_evaluado':float(xx),'y_estimado':float(yy)})

        if m == 'cubic':
            deg = 3
            if len(xs) < 4:
                note = 'Pocos datos para grado 3; se usa grado disponible.'
                deg = min(3, len(xs)-1)
            coeffs = np.polyfit(xs, ys, deg)
            y_plot = poly_predict_from_coeffs(coeffs, x_plot)
            y_at_xs = poly_predict_from_coeffs(coeffs, xs)
            overlay_traces.append(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Cúbica'))
            x_table = np.array(xq) if xq else xs
            y_table = poly_predict_from_coeffs(coeffs, x_table)
            results.append({'method':'Cúbica','rmse':round(rmse(ys,y_at_xs),8),'mae':round(mae(ys,y_at_xs),8),'maxerr':round(max_error(ys,y_at_xs),8),'r2':round(r2_score(ys,y_at_xs),8),'note':note})
            if plot_mode in ('all','individuals'):
                block, df_tab = add_individual_block('Cúbica', x_plot, y_plot, x_table, y_table)
                individual_children.append(block)
            for xx, yy in zip(x_table, y_table):
                combined_rows.append({'method':'Cúbica','x_evaluado':float(xx),'y_estimado':float(yy)})

        if m.startswith('lag'):
            deg = int(m.replace('lag',''))
            f = lagrange_evaluator_degree(xs, ys, deg)
            y_plot = f(x_plot)
            y_at_xs = f(xs)
            overlay_traces.append(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f'Lagrange deg{deg}'))
            x_table = np.array(xq) if xq else xs
            y_table = f(x_table)
            results.append({'method':f'Lagrange deg{deg}','rmse':round(rmse(ys,y_at_xs),8),'mae':round(mae(ys,y_at_xs),8),'maxerr':round(max_error(ys,y_at_xs),8),'r2':round(r2_score(ys,y_at_xs),8),'note':note})
            if plot_mode in ('all','individuals'):
                block, df_tab = add_individual_block(f'Lagrange deg{deg}', x_plot, y_plot, x_table, y_table)
                individual_children.append(block)
            for xx, yy in zip(x_table, y_table):
                combined_rows.append({'method':f'Lagrange deg{deg}','x_evaluado':float(xx),'y_estimado':float(yy)})

        if m.startswith('new'):
            deg = int(m.replace('new',''))
            f = newton_divided_coeffs(xs, ys, deg)
            y_plot = f(x_plot)
            y_at_xs = f(xs)
            overlay_traces.append(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f'Newton deg{deg}'))
            x_table = np.array(xq) if xq else xs
            y_table = f(x_table)
            results.append({'method':f'Newton deg{deg}','rmse':round(rmse(ys,y_at_xs),8),'mae':round(mae(ys,y_at_xs),8),'maxerr':round(max_error(ys,y_at_xs),8),'r2':round(r2_score(ys,y_at_xs),8),'note':note})
            if plot_mode in ('all','individuals'):
                block, df_tab = add_individual_block(f'Newton deg{deg}', x_plot, y_plot, x_table, y_table)
                individual_children.append(block)
            for xx, yy in zip(x_table, y_table):
                combined_rows.append({'method':f'Newton deg{deg}','x_evaluado':float(xx),'y_estimado':float(yy)})

        if m == 'inv3':
            deg = 3
            if len(xs) < deg+1:
                note = 'Pocos datos para deg3; se usa grado disponible.'
                deg = min(len(xs)-1, deg)
            coeffs = np.polyfit(xs, ys, deg)
            y_plot = poly_predict_from_coeffs(coeffs, x_plot)
            overlay_traces.append(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f'Inv poly deg{deg}'))
            # inversa: solo para y_targets
            if len(yts)>0:
                x_solutions, _ = inverse_interpolation_polynomial(xs, ys, deg, yts)
                for yt, xsln in zip(yts, x_solutions):
                    combined_rows.append({'method':f'Inv poly deg{deg}','x_evaluado':float(xsln) if not np.isnan(xsln) else np.nan,'y_estimado':float(yt)})
                # individual block: show only y_targets -> x_results
                df_inv = pd.DataFrame({'y_objetivo': yts, 'x_resultado': list(x_solutions)})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=f'Inv poly deg{deg}'))
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name='Datos', marker={'size':6,'color':'black'}))
                fig.update_layout(title=f'Interpolación inversa (deg {deg})', margin={'t':30}, height=300)
                datatable = dash_table.DataTable(
                    columns=[{"name":"y_objetivo","id":"y_objetivo"},{"name":"x_resultado","id":"x_resultado"}],
                    data=df_inv.to_dict('records'),
                    style_cell={'textAlign':'center','padding':'4px','fontSize':'12px'},
                    style_header={'backgroundColor':'#f1f1f1','fontWeight':'bold'},
                    page_size=6
                )
                block = html.Div(style={'marginBottom':'12px','paddingBottom':'8px','borderBottom':'1px solid #eee'}, children=[
                    dcc.Graph(figure=fig, style={'height':'300px'}),
                    html.Div(datatable)
                ])
                if plot_mode in ('all','individuals'):
                    individual_children.append(block)
            # compute forward predictions at xs for errors
            y_at_xs = poly_predict_from_coeffs(coeffs, xs)
            results.append({'method':f'Inv poly deg{deg}','rmse':round(rmse(ys,y_at_xs),8),'mae':round(mae(ys,y_at_xs),8),'maxerr':round(max_error(ys,y_at_xs),8),'r2':round(r2_score(ys,y_at_xs),8),'note':note})

    # overlay figure
    fig_overlay = go.Figure(data=overlay_traces)
    fig_overlay.update_layout(title='Overlay de métodos', xaxis={'title':'x'}, yaxis={'title':'y'}, margin={'t':30, 'b':20}, height=380)

    # combined table component (right panel)
    if len(combined_rows) > 0:
        df_comb = pd.DataFrame(combined_rows)
        df_comb = df_comb.sort_values(['method','x_evaluado'], na_position='last').reset_index(drop=True)
        combined_table = dash_table.DataTable(
            columns=[{"name":"Método","id":"method"},{"name":"x_evaluado","id":"x_evaluado"},{"name":"y_estimado","id":"y_estimado"}],
            data=df_comb.to_dict('records'),
            page_size=8,
            style_cell={'textAlign':'center','padding':'4px','fontSize':'12px'},
            style_header={'backgroundColor':'#f1f1f1','fontWeight':'bold'},
            style_table={'overflowX':'auto'}
        )
    else:
        combined_table = html.Div("No hay valores calculados para mostrar.")

    # prepare results (errors) as list-of-dicts for DataTable and store them + combined rows in last-run-store
    df_results = pd.DataFrame(results) if len(results)>0 else pd.DataFrame(columns=['method','rmse','mae','maxerr','r2','note'])
    store_payload = {
        'results': df_results.to_dict('records'),
        'combined': pd.DataFrame(combined_rows).to_dict('records') if len(combined_rows)>0 else []
    }

    return fig_overlay, df_results.to_dict('records'), individual_children, combined_table, store_payload

# ----------------- CALLBACK: export CSV usando last-run-store -----------------

@app.callback(
    Output("about-modal", "children"),
    Input("open-about", "n_clicks"),
    Input("close-about", "n_clicks"),
    prevent_initial_call=True
)
def toggle_about(open_click, close_click):

    # si se presiona “Cerrar”
    if dash.callback_context.triggered_id == "close-about":
        return ""

    # si se presiona “Acerca de”
    about_text = """
    Acerca de este software

    Nombre del software: Sistema de Interpolación y Extrapolación – Proyecto Final ATN
    Autores: Amaurys Castro De Arco, Daniel Jimenez Salcedo
    Asignatura: Análisis de Técnicas Numéricas
    Programa académico: Ingeniería de Sistemas
    Universidad: Corporación Universitaria Del Caribe - CECAR
    Año: 2025

    Descripción general

    Este software fue desarrollado como proyecto final del curso Análisis de Técnicas Numéricas, con el objetivo de implementar de manera computacional los métodos de interpolación y extrapolación estudiados en clase. La herramienta permite cargar datos reales desde archivos CSV, aplicar múltiples métodos numéricos y visualizar los resultados de manera clara mediante gráficas y tablas comparativas.

    Métodos incluidos

    El sistema implementa los métodos vistos en clase:

* Interpolación

* Interpolación lineal

* Interpolación polinomial cuadrática

* Interpolación polinomial cúbica

* Polinomios de Lagrange grados 1, 2 y 3

* Polinomios de Newton (diferencias divididas) grados 1, 2 y 3

* Interpolación inversa mediante polinomio de grado 3

* Extrapolación

Realizada automáticamente por los polinomios anteriores cuando se evalúan puntos fuera del rango.

Características principales

Carga de archivos CSV con columnas x y y.

Ejecución individual o combinada de todos los métodos.

Gráficas individuales por método y gráfica general comparativa.

Tablas de resultados por método (x evaluados y valores estimados).

Tabla combinada para comparar todos los métodos.

Cálculo automático de métricas de error:

RMSE

MAE

Error Máximo (MaxErr)

Coeficiente de determinación (R²)

Tecnologías utilizadas

Lenguaje: Python 3

Librerías: Dash, Plotly, NumPy, Pandas

Arquitectura tipo Single Page Application (SPA)

Guía básica de uso

Cargue un archivo CSV con columnas x y y.

Seleccione los métodos de interpolación deseados.

Ingrese valores de x para evaluar (opcional).

Ingrese valores de y para interpolación inversa (opcional).

Presione “Ejecutar métodos”.

Revise:

Gráfica comparativa general

Gráficas individuales

Tablas individuales

Tabla comparada con todos los métodos

Métricas de error

Licencia

Este software es de uso académico y no está destinado a uso comercial.
    """

    return html.Div(
        style={
            'backgroundColor': 'rgba(0,0,0,0.6)',
            'position': 'fixed','top': 0,'left': 0,
            'width': '100%','height': '100%',
            'display': 'flex','alignItems': 'center',
            'justifyContent': 'center','zIndex': 9999
        },
        children=[
            html.Div(
                style={
                    'backgroundColor':'white','padding':'20px',
                    'borderRadius':'8px','width':'50%',
                    'boxShadow':'0 0 10px rgba(0,0,0,0.3)',
                    'textAlign':'left'
                },
                children=[
                    html.H2("Acerca de"),
                    html.Pre(about_text, style={'whiteSpace':'pre-wrap'}),
                    html.Button("Cerrar", id="close-about", style={
                        'marginTop':'10px','padding':'8px',
                        'background':'#c0392b','color':'white',
                        'border':'none','borderRadius':'5px'
                    })
                ]
            )
        ]
    )


# ----------------- RUN -----------------
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8080, debug=False)
    #app.run(debug=True)
