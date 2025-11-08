"""
Web interface for the REACT agent with interactive map and chat.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
from orchestrator import AgriAgent, REACTAgent, Tool

app = Flask(__name__)
CORS(app)

# Global agent instance
agent = None


def initialize_agent():
    """Initialize the AgriAgent with API key."""
    global agent
    try:
        with open("OPENROUTER_API_KEY.txt", "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise Exception("OPENROUTER_API_KEY.txt not found")
    
    agent = AgriAgent(api_key=api_key)
    
    return agent


@app.route('/')
def index():
    """Serve the main interface page."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and return agent response."""
    global agent
    
    if agent is None:
        try:
            initialize_agent()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Run the agent with the user's message
        result = agent.run(message)
        
        return jsonify({
            'response': result,
            'iterations': agent.iteration_count,
            'observations': agent.observation_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/map/bounds', methods=['POST'])
def update_map_bounds():
    """Handle map bounds updates (for future use)."""
    data = request.json
    # Store or process map bounds if needed
    return jsonify({'status': 'ok'})


@app.route('/api/bounding-box/point', methods=['POST'])
def add_bounding_point():
    """Add a single bounding point to the agent."""
    global agent
    
    if agent is None:
        try:
            initialize_agent()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    
    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng required'}), 400
    
    try:
        point = (float(lat), float(lng))
        agent.add_bounding_point(point)
        return jsonify({
            'status': 'success',
            'point': point,
            'total_points': len(agent.bounding_points),
            'message': f'Point added. Total points: {len(agent.bounding_points)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bounding-box/clear', methods=['POST'])
def clear_bounding_points():
    """Clear all bounding points from the agent."""
    global agent
    
    if agent is None:
        try:
            initialize_agent()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        count = len(agent.bounding_points)
        agent.bounding_points = []
        return jsonify({
            'status': 'success',
            'message': f'Cleared {count} points'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bounding-box', methods=['GET'])
def get_bounding_box():
    """Get current bounding box points from the agent."""
    global agent
    
    if agent is None:
        try:
            initialize_agent()
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({
        'points': agent.bounding_points
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, port=10000)

