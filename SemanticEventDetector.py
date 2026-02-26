# SemanticEventDetector.py - Detects and annotates semantic events

class SemanticEventDetector:
    """Detects meaningful events from agent actions and state changes"""
    
    def __init__(self):
        self.action_names = {
            0: "noop", 1: "move_left", 2: "move_right", 3: "move_up", 4: "move_down",
            5: "do", 6: "sleep", 7: "place_stone", 8: "place_table", 9: "place_furnace",
            10: "place_plant", 11: "make_wood_pickaxe", 12: "make_stone_pickaxe",
            13: "make_iron_pickaxe", 14: "make_wood_sword", 15: "make_stone_sword",
            16: "make_iron_sword"
        }
        
        # Event templates for human-readable descriptions
        self.event_templates = {
            'resource_collected': "🌲 Collected {resource}",
            'item_crafted': "🔨 Crafted {item}",
            'structure_built': "🏗️ Built {structure}",
            'enemy_defeated': "⚔️ Defeated {enemy}",
            'exploration': "🗺️ Explored new area",
            'survival': "💤 Rested to recover health",
            'milestone': "🎯 Achieved {milestone}"
        }
    
    def detect_events(self, time_steps, actions, reward_components):
        """Detect semantic events from the trajectory data"""
        events = []
        
        for i, step in enumerate(time_steps):
            step_events = []
            
            # Check for resource collection events
            if reward_components:
                for component, values in reward_components.items():
                    if i < len(values) and values[i] > 0:
                        if component.startswith('collect_'):
                            resource = component.replace('collect_', '').replace('_', ' ').title()
                            step_events.append({
                                'type': 'resource_collected',
                                'description': self.event_templates['resource_collected'].format(resource=resource),
                                'importance': 'medium',
                                'component': component,
                                'value': values[i]
                            })
                        elif component.startswith('make_'):
                            item = component.replace('make_', '').replace('_', ' ').title()
                            step_events.append({
                                'type': 'item_crafted',
                                'description': self.event_templates['item_crafted'].format(item=item),
                                'importance': 'high',
                                'component': component,
                                'value': values[i]
                            })
                        elif component.startswith('place_'):
                            structure = component.replace('place_', '').replace('_', ' ').title()
                            step_events.append({
                                'type': 'structure_built',
                                'description': self.event_templates['structure_built'].format(structure=structure),
                                'importance': 'high',
                                'component': component,
                                'value': values[i]
                            })
                        elif component.startswith('defeat_'):
                            enemy = component.replace('defeat_', '').replace('_', ' ').title()
                            step_events.append({
                                'type': 'enemy_defeated',
                                'description': self.event_templates['enemy_defeated'].format(enemy=enemy),
                                'importance': 'high',
                                'component': component,
                                'value': values[i]
                            })
                        elif component == 'wake_up':
                            step_events.append({
                                'type': 'survival',
                                'description': self.event_templates['survival'],
                                'importance': 'medium',
                                'component': component,
                                'value': values[i]
                            })
            
            # Check for action-based events
            if i < len(actions):
                action = actions[i]
                action_name = self.action_names.get(action, f"unknown_{action}")
                
                # Detect exploration (movement actions)
                if action_name in ['move_left', 'move_right', 'move_up', 'move_down']:
                    # Only mark as exploration if it's the first movement in a sequence
                    if i == 0 or actions[i-1] not in [1, 2, 3, 4]:
                        step_events.append({
                            'type': 'exploration',
                            'description': self.event_templates['exploration'],
                            'importance': 'low',
                            'component': 'movement',
                            'value': 1
                        })
            
            # Add events for this step
            if step_events:
                events.append({
                    'step': step,
                    'events': step_events
                })
        
        return events
    
    def get_event_color(self, event_type):
        """Get color for event type"""
        colors = {
            'resource_collected': '#2E8B57',  # Sea Green
            'item_crafted': '#FF6347',        # Tomato
            'structure_built': '#4682B4',     # Steel Blue
            'enemy_defeated': '#DC143C',      # Crimson
            'exploration': '#DAA520',         # Goldenrod
            'survival': '#9370DB',            # Medium Purple
            'milestone': '#FF1493'            # Deep Pink
        }
        return colors.get(event_type, '#708090')  # Slate Gray default
