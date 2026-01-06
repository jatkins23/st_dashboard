"""Minimal Dash app for testing."""

from dash import Dash, html
import dash_bootstrap_components as dbc

# Create a minimal Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY]
)

# Set a simple layout
app.layout = dbc.Container([
    dbc.Alert("Minimal Dash App is Running!", color="success"),
    html.H1("Hello World"),
    html.P("If you can see this, Dash is working correctly.")
], className="p-5")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Minimal Dash Test App")
    print("="*60)
    print("URL: http://127.0.0.1:8053")
    print("="*60 + "\n")

    app.run(debug=False, host='127.0.0.1', port=8053)
