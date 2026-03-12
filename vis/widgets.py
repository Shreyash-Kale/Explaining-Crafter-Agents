# Vis.py - Updated with improved colors, cleaner annotations, and better styling

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                           QGraphicsDropShadowEffect, QSizePolicy, QGridLayout, QPushButton, 
                           QTabWidget, QListWidget, QStyle, QFrame, QGraphicsDropShadowEffect, QListWidgetItem,
                           QScrollArea,
                           QToolTip)
from PyQt5.QtCore import Qt, pyqtSlot, QRectF, QEvent
from PyQt5.QtGui import QColor, QPainter, QFont, QBrush, QPicture, QIcon, QCursor
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from .explainer import generate_explanation


class InfoPanel(QFrame):
    """Panel that displays achievement information in two tabs"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up styling
        self.setFrameShape(QFrame.StyledPanel)
        self.data_manager = None
        
        # Create layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(5)
        
        # Title with proper contrast
        self.title_label = QLabel("Agent Achievements")
        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #111111;
            background-color: #f0f0f0;
            padding: 3px;
            border-radius: 3px;
        """)
        self.main_layout.addWidget(self.title_label)
        
        # Create tabbed container for achievements
        self.achievements_tabs = QTabWidget()
        self.achievements_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
                margin-right: 2px;
                color: #666666;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
                color: #111111;
            }
        """)
        
        # Create tab for completed achievements
        self.completed_tab = QWidget()
        completed_layout = QVBoxLayout(self.completed_tab)
        completed_layout.setContentsMargins(5, 5, 5, 5)
        completed_layout.setSpacing(2)
        
        self.completed_list = QListWidget()
        self.completed_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.completed_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e0e0e0;
                color: #006600;
            }
            QListWidget::item:hover {
                background-color: #f0fff0;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #cccccc;
                min-height: 20px;
                border-radius: 5px;
            }
        """)
        completed_layout.addWidget(self.completed_list)
        
        # Create tab for available achievements
        self.available_tab = QWidget()
        available_layout = QVBoxLayout(self.available_tab)
        available_layout.setContentsMargins(5, 5, 5, 5)
        available_layout.setSpacing(2)
        
        self.available_list = QListWidget()
        self.available_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.available_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e0e0e0;
                color: #666666;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #cccccc;
                min-height: 20px;
                border-radius: 5px;
            }
        """)
        available_layout.addWidget(self.available_list)
        
        # Add tabs to container
        self.achievements_tabs.addTab(self.completed_tab, "Completed (0)")
        self.achievements_tabs.addTab(self.available_tab, "Available (22)")
        
        # Add the tabs widget to the main layout
        self.main_layout.addWidget(self.achievements_tabs)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        # Initialize achievement lists
        self.initialize_achievements()
    
    def initialize_achievements(self):
        """Initialize the achievement lists - all start in available tab"""
        self.achievement_list = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        # Initially all achievements are in the available tab
        self.achievements_tabs.addTab(self.completed_tab, "Completed (0)")
        self.achievements_tabs.addTab(self.available_tab, f"Available ({len(self.achievement_list)})")

        
        # Add all achievements to the available list
        for ach in self.achievement_list:
            item = QListWidgetItem()
            name = ach.replace('_', ' ').title()
            
            # Check if dependencies exist to show additional info
            if self.data_manager:
                deps = self.data_manager.get_achievement_dependencies(ach)
                if deps:
                    deps_text = ", ".join(d.replace('_', ' ').title() for d in deps)
                    item.setText(f"◯ {name} (Needs: {deps_text})")
                else:
                    item.setText(f"◯ {name}")
            else:
                item.setText(f"◯ {name}")
                
            item.setData(Qt.UserRole, ach)  # Store achievement ID
            self.available_list.addItem(item)
    
    
    def update_state(self, step_data):
        """Update the achievements based on the current timeline position"""
        if not step_data or not self.data_manager:
            return
            
         # Check if step_data is an integer or a dictionary
        if isinstance(step_data, dict):
            current_step = step_data.get('time_step', 0)
        else:
            # If it's an integer, use it directly as the step
            current_step = step_data
        
        # Get completed achievements UP TO the current step
        completed = self.data_manager.get_completed_achievements(step=current_step)
        step_achievements = self.data_manager.get_step_achievements(current_step)
        
        # Track which achievements need to move between tabs
        to_move_to_completed = []
        to_move_to_available = []
        
        # Check available achievements to see if any should move to completed
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            ach_id = item.data(Qt.UserRole)
            
            if ach_id in completed:
                to_move_to_completed.append(ach_id)
        
        # Check completed achievements to see if any should move back to available
        for i in range(self.completed_list.count()):
            item = self.completed_list.item(i)
            ach_id = item.data(Qt.UserRole)
            
            if ach_id not in completed:
                to_move_to_available.append(ach_id)
        
        # Move items between lists
        for ach_id in to_move_to_completed:
            self._move_achievement_to_completed(ach_id, ach_id in step_achievements)
            
        for ach_id in to_move_to_available:
            self._move_achievement_to_available(ach_id)
        
        # Update available tab to show dependencies
        self._update_available_dependencies(completed)
        
        # Update tab titles with counts
        self.achievements_tabs.setTabText(0, f"Completed ({self.completed_list.count()})")
        self.achievements_tabs.setTabText(1, f"Available ({self.available_list.count()})")
        
        # Update main title with counts
        total = len(self.achievement_list)
        completed_count = self.completed_list.count()
        self.title_label.setText(f"Agent Achievements ({completed_count}/{total})")
    
    def _move_achievement_to_completed(self, ach_id, is_new=False):
        """Move an achievement from available to completed tab"""
        # Find and remove from available list
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            if item.data(Qt.UserRole) == ach_id:
                item = self.available_list.takeItem(i)
                name = ach_id.replace('_', ' ').title()
                
                # Modify the item for completed list
                if is_new:
                    item.setBackground(QBrush(QColor('#7fff00')))  # Chartreuse green
                    item.setForeground(QBrush(QColor('#000000')))  # Black text for contrast
                    # Make font bold for new achievements
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                else:
                    item.setText(f"✓ {name}")
                    item.setBackground(QBrush(QColor('#b3e6cc')))  # Light green
                    item.setForeground(QBrush(QColor('#006600')))  # Dark green text
                
                # Add to completed list
                self.completed_list.addItem(item)
                break
    
    def _move_achievement_to_available(self, ach_id):
        """Move an achievement from completed back to available tab"""
        # Find and remove from completed list
        for i in range(self.completed_list.count()):
            item = self.completed_list.item(i)
            if item.data(Qt.UserRole) == ach_id:
                item = self.completed_list.takeItem(i)
                name = ach_id.replace('_', ' ').title()
                
                # Reset formatting
                item.setBackground(QBrush(QColor('#ffffff')))  # White
                item.setForeground(QBrush(QColor('#666666')))  # Gray text
                font = item.font()
                font.setBold(False)
                item.setFont(font)
                
                # Check if dependencies exist
                if self.data_manager:
                    deps = self.data_manager.get_achievement_dependencies(ach_id)
                    if deps:
                        deps_text = ", ".join(d.replace('_', ' ').title() for d in deps)
                        item.setText(f"◯ {name} (Needs: {deps_text})")
                    else:
                        item.setText(f"◯ {name}")
                else:
                    item.setText(f"◯ {name}")
                
                # Add to available list
                self.available_list.addItem(item)
                break
    
    def _update_available_dependencies(self, completed):
        """Update the status of available achievements based on dependencies"""
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            ach_id = item.data(Qt.UserRole)
            name = ach_id.replace('_', ' ').title()
            
            if self.data_manager:
                deps = self.data_manager.get_achievement_dependencies(ach_id)
                if deps:
                    # Check if all dependencies are completed
                    deps_completed = all(d in completed for d in deps)
                    if deps_completed:
                        # Achievement is ready to be completed
                        item.setText(f"◯ {name} (Ready!)")
                        item.setBackground(QBrush(QColor('#fff8e1')))  # Light amber
                        item.setForeground(QBrush(QColor('#996600')))  # Dark amber text
                    else:
                        # Show which dependencies are needed
                        deps_text = ", ".join(d.replace('_', ' ').title() for d in deps)
                        item.setText(f"◯ {name} (Needs: {deps_text})")
                        item.setBackground(QBrush(QColor('#ffffff')))  # White
                        item.setForeground(QBrush(QColor('#666666')))  # Gray text


class ExplanationPanel(QFrame):
    """Bottom toolbox that narrates agent decisions in natural language."""

    def __init__(self, data_manager=None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.current_step = 0
        self.algorithm = "unknown"

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #d7d7d7; border-radius: 6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self.title_label = QLabel("Explanation Toolbox")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #1f1f1f;")
        layout.addWidget(self.title_label)

        self.meta_label = QLabel("Waiting for trajectory data...")
        self.meta_label.setStyleSheet("font-size: 11px; color: #5a5a5a;")
        layout.addWidget(self.meta_label)

        self.explanation_label = QLabel(
            "Load a trajectory and scrub the timeline to see per-step explanations."
        )
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.explanation_label.setStyleSheet(
            "font-size: 12px; color: #222; line-height: 1.35; "
            "background-color: #fafafa; border: 1px solid #ececec; border-radius: 4px; padding: 8px;"
        )
        layout.addWidget(self.explanation_label, 1)

    def set_data_manager(self, data_manager):
        self.data_manager = data_manager
        self.update_step(0)

    def set_algorithm(self, algorithm):
        self.algorithm = (algorithm or "unknown").lower()
        self.update_step(self.current_step)

    def update_step(self, step):
        """Refresh explanation text using current and previous timestep rows."""
        self.current_step = max(0, int(step)) if step is not None else 0
        if self.data_manager is None or self.data_manager.event_df is None:
            return

        max_idx = len(self.data_manager.event_df) - 1
        if max_idx < 0:
            return

        idx = min(self.current_step, max_idx)
        curr = self.data_manager.get_step_details(idx)
        prev = self.data_manager.get_step_details(max(0, idx - 1))
        if not curr:
            return

        if self.algorithm in ("unknown", ""):
            if 'world_model_score' in curr and 'exploration_bonus' in curr:
                algo = 'dreamer'
            elif 'entropy' in curr and 'advantage' in curr:
                algo = 'ppo'
            else:
                algo = 'unknown'
        else:
            algo = self.algorithm

        step_achievements = self.data_manager.get_step_achievements(idx)
        if step_achievements:
            curr = dict(curr)
            curr['achievement_unlocked'] = step_achievements[0]

        text = generate_explanation(curr, prev_row=prev, algorithm=algo)
        action_name = self.data_manager.get_action_name(curr.get('action'))
        reward = curr.get('reward', 0.0)

        self.meta_label.setText(
            f"Step {idx} | Action: {action_name} | Reward: {float(reward):.3f} | Algorithm: {algo.upper()}"
        )
        self.explanation_label.setText(text)


class CustomBarGraphItem(pg.GraphicsObject):
    """Custom bar graph with improved styling and tooltips"""
    
    def __init__(self, x, height, width=0.8, brushes=None, pens=None):
        pg.GraphicsObject.__init__(self)
        self.x = np.array(x)
        self.height = np.array(height)
        self.width = width

        if brushes is None:
            self.brushes = [pg.mkBrush(100, 100, 255, 150) for _ in height]
        else:
            self.brushes = brushes
            
        if pens is None:
            self.pens = [pg.mkPen(None) for _ in height]
        else:
            self.pens = pens
            
        self._picture = None
        self._boundingRect = None
        self.generatePicture()
    
    def generatePicture(self):
        """Pre-render the bars as a QPicture object"""
        self._picture = QPicture()
        painter = QPainter(self._picture)
        
        for i in range(len(self.x)):
            x, h = self.x[i], self.height[i]
            
            if h == 0:  # Skip zero-height bars
                continue
                
            rect = QRectF(x - self.width/2, 0, self.width, h)
            painter.setBrush(self.brushes[i])
            painter.setPen(self.pens[i])
            
            # Draw rectangle with rounded corners for positive values
            if h > 0:
                painter.drawRoundedRect(rect, 2, 2)
            else:
                painter.drawRect(rect)
                
        painter.end()
        
        # Calculate bounding rect
        xmin = min(self.x) - self.width/2
        xmax = max(self.x) + self.width/2
        ymin = min(0, min(self.height))
        ymax = max(0, max(self.height))
        
        self._boundingRect = QRectF(xmin, ymin, xmax-xmin, ymax-ymin)
    
    def paint(self, painter, option, widget):
        painter.drawPicture(0, 0, self._picture)
    
    def boundingRect(self):
        return self._boundingRect


class DecisionPoint(pg.ScatterPlotItem):
    """Enhanced scatter plot item that highlights decision points"""
    
    def __init__(self, x, y, decision_type, importance, actions=None, **kwargs):
        self.decision_type = decision_type  # e.g., 'positive', 'negative', 'neutral'
        self.importance = importance  # Numeric importance (determines size)
        self.actions = actions if actions else []
        
        # Determine symbol based on decision type
        symbol = 'o'  # Default
        if decision_type == 'positive':
            symbol = 't'  # Triangle
        elif decision_type == 'negative':
            symbol = 'd'  # Diamond
        
        # Determine size based on importance
        size = 10 + (importance * 5)
        
        # Determine color based on decision type
        brush = pg.mkBrush(50, 50, 200, 200)  # Default blue
        if decision_type == 'positive':
            brush = pg.mkBrush(50, 200, 50, 200)  # Green
        elif decision_type == 'negative':
            brush = pg.mkBrush(200, 50, 50, 200)  # Red
        
        # Call parent constructor with calculated properties
        super().__init__(x=x, y=y, size=size, symbol=symbol, brush=brush, **kwargs)


class LegendToggleRow(QFrame):
    """Clickable external legend row that toggles plot visibility."""

    def __init__(self, label_text, color_css, toggle_callback, parent=None):
        super().__init__(parent)
        self.toggle_callback = toggle_callback
        self.color_css = color_css
        self.is_active = True

        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.swatch = QLabel()
        self.swatch.setFixedSize(12, 12)
        layout.addWidget(self.swatch)

        self.label = QLabel(label_text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)

        self._refresh_style()

    def mousePressEvent(self, event):
        self.set_active(not self.is_active)
        if self.toggle_callback is not None:
            self.toggle_callback(self.is_active)
        super().mousePressEvent(event)

    def set_active(self, is_active):
        self.is_active = is_active
        self._refresh_style()

    def _refresh_style(self):
        label_color = '#222' if self.is_active else '#999'
        border_color = '#666' if self.is_active else '#bbb'
        self.label.setStyleSheet(f"font-size: 10px; color: {label_color};")
        self.swatch.setStyleSheet(
            f"background: {self.color_css}; border: 1px solid {border_color}; border-radius: 2px;"
        )

class DecisionAttribPlot(QWidget):
    """Decision attribution plot with external right-side legend panel."""

    def __init__(self, dm, hover_step_callback=None):
        super().__init__()
        self.dm = dm
        self.hover_step_callback = hover_step_callback
        self.curve_data = []
        self.hover_text = None
        self.hover_line = None
        self.smoothing_window = 7

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Normalised Value')
        self.plot.setLabel('bottom', 'Time Step')
        self.plot.getAxis('left').setTextPen('k')
        self.plot.getAxis('bottom').setTextPen('k')
        self.plot.getViewBox().setDefaultPadding(0.0)
        self.plot.getViewBox().setLimits(xMin=0)
        self.plot.installEventFilter(self)
        root.addWidget(self.plot, 1)

        self._legend_inner = QWidget()
        self._legend_inner.setStyleSheet("background: white; border-left: 1px solid #ddd;")
        self._legend_vbox = QVBoxLayout(self._legend_inner)
        self._legend_vbox.setContentsMargins(8, 8, 8, 8)
        self._legend_vbox.setSpacing(5)
        title = QLabel("Decision Attribution Legend")
        title.setStyleSheet("font-weight: bold; font-size: 10px; color: #333;")
        self._legend_vbox.addWidget(title)
        self._legend_vbox.addStretch(1)
        self.curves = {}

        legend_scroll = QScrollArea()
        legend_scroll.setWidget(self._legend_inner)
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setFixedWidth(170)
        legend_scroll.setFrameShape(QFrame.NoFrame)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(legend_scroll)

        self.hover_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.on_hover,
        )

        self._build_plot()

    def _smooth_series(self, values):
        """Light moving-average smoothing to reduce decision-trace jitter."""
        arr = np.array(values, dtype=float)
        if arr.size < 3:
            return arr

        # Keep an odd window and avoid oversmoothing short traces.
        window = min(self.smoothing_window, arr.size if arr.size % 2 == 1 else arr.size - 1)
        window = max(3, window)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return arr

        kernel = np.ones(window, dtype=float) / float(window)
        pad = window // 2
        padded = np.pad(arr, (pad, pad), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    def _add_legend_row(self, name, color_hex, curve):
        row = LegendToggleRow(name, color_hex, lambda visible, item=curve: item.setVisible(visible))
        self._legend_vbox.insertWidget(self._legend_vbox.count() - 1, row)

    def _build_plot(self):
        x = np.arange(len(self.dm.time_steps))

        curves = [
            ("Value estimate", self.dm.get_value_norm(), '#9467bd', pg.mkPen('#9467bd', width=2)),
            ("Action probability", self.dm.get_action_prob_norm(), '#8c564b', pg.mkPen('#8c564b', width=2, style=Qt.DashLine)),
        ]

        ppo_ok = (len(self.dm.get_ppo_entropy_norm()) == len(x) and np.any(self.dm.get_ppo_entropy_norm()))
        dr_ok = (len(self.dm.get_dreamer_explore_norm()) == len(x) and np.any(self.dm.get_dreamer_explore_norm()))

        if ppo_ok:
            curves += [
                ("PPO entropy", self.dm.get_ppo_entropy_norm(), '#1f77b4', pg.mkPen('#1f77b4', width=2)),
                ("PPO advantage", self.dm.get_ppo_advantage_norm(), '#d62728', pg.mkPen('#d62728', width=2, style=Qt.DashLine)),
            ]
        elif dr_ok:
            curves += [
                ("Exploration bonus", self.dm.get_dreamer_explore_norm(), '#ff7f0e', pg.mkPen('#ff7f0e', width=2)),
                ("World-model score", self.dm.get_dreamer_wm_score_norm(), '#2ca02c', pg.mkPen('#2ca02c', width=2, style=Qt.DashLine)),
            ]

        for name, y, color_hex, pen in curves:
            if len(y) == len(x) and np.any(y):
                smooth_y = self._smooth_series(y)
                curve = self.plot.plot(x, smooth_y, pen=pen, name=None)
                self.curves[name] = curve
                self.curve_data.append({
                    'name': name,
                    'values': np.array(smooth_y, dtype=float),
                    'color_hex': color_hex,
                    'curve': curve,
                })
                self._add_legend_row(name, color_hex, curve)

        algo = "PPO" if ppo_ok else "Dreamer" if dr_ok else "Agent"
        self.plot.setTitle(f"Decision Attribution – {algo}", color="#333", size="12pt")

        if len(x) > 1:
            self.plot.setXRange(0, len(x) - 1, padding=0)

        self.marker = pg.InfiniteLine(0, angle=90, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        self.plot.addItem(self.marker)
        self.external_hover_line = pg.InfiniteLine(0, angle=90, pen=pg.mkPen(color=(80, 80, 80), width=1, style=Qt.DotLine))
        self.external_hover_line.setVisible(False)
        self.plot.addItem(self.external_hover_line)

    def update_marker(self, idx):
        self.marker.setPos(idx)

    def set_external_hover_idx(self, idx):
        self.external_hover_line.setPos(idx)
        self.external_hover_line.setVisible(True)

    def clear_external_hover(self):
        self.external_hover_line.setVisible(False)

    def eventFilter(self, obj, event):
        if obj == self.plot and event.type() in (QEvent.Leave, QEvent.HoverLeave):
            self.clear_hover_items()
            if self.hover_step_callback is not None:
                self.hover_step_callback(None, source='decision')
        return super().eventFilter(obj, event)

    def clear_hover_items(self):
        if self.hover_text is not None:
            try:
                self.plot.removeItem(self.hover_text)
            except Exception:
                pass
            self.hover_text = None
        if self.hover_line is not None:
            try:
                self.plot.removeItem(self.hover_line)
            except Exception:
                pass
            self.hover_line = None

    def _place_tooltip(self, text_item, point_x, point_y):
        vb = self.plot.getViewBox()
        view_rect = self.plot.viewRect()
        scene_rect = vb.sceneBoundingRect()
        data_per_px_x = view_rect.width() / max(scene_rect.width(), 1)
        data_per_px_y = view_rect.height() / max(scene_rect.height(), 1)
        text_rect = text_item.boundingRect()
        tip_w = text_rect.width() * data_per_px_x
        tip_h = text_rect.height() * data_per_px_y

        x_margin = data_per_px_x * 10
        y_margin = data_per_px_y * 8

        if point_x + x_margin + tip_w <= view_rect.right():
            anchor_x = 0
            x_pos = point_x + x_margin
        else:
            anchor_x = 1
            x_pos = point_x - x_margin

        if point_y + y_margin + tip_h <= view_rect.top():
            anchor_y = 0
            y_pos = point_y + y_margin
        elif point_y - y_margin - tip_h >= view_rect.bottom():
            anchor_y = 1
            y_pos = point_y - y_margin
        else:
            if (view_rect.top() - point_y) >= (point_y - view_rect.bottom()):
                anchor_y = 0
                y_pos = max(view_rect.bottom() + tip_h, min(point_y + y_margin, view_rect.top() - tip_h))
            else:
                anchor_y = 1
                y_pos = min(view_rect.top() - tip_h, max(point_y - y_margin, view_rect.bottom() + tip_h))

        text_item.setAnchor((anchor_x, anchor_y))
        text_item.setPos(x_pos, y_pos)

        item_rect = text_item.mapRectToParent(text_item.boundingRect())
        dx = 0
        dy = 0
        if item_rect.left() < view_rect.left():
            dx = view_rect.left() - item_rect.left()
        elif item_rect.right() > view_rect.right():
            dx = view_rect.right() - item_rect.right()
        if item_rect.bottom() < view_rect.bottom():
            dy = view_rect.bottom() - item_rect.bottom()
        elif item_rect.top() > view_rect.top():
            dy = view_rect.top() - item_rect.top()
        if dx or dy:
            text_item.setPos(text_item.pos().x() + dx, text_item.pos().y() + dy)

    def on_hover(self, event):
        self.clear_hover_items()

        if not self.curve_data:
            if self.hover_step_callback is not None:
                self.hover_step_callback(None, source='decision')
            return

        pos = event[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            if self.hover_step_callback is not None:
                self.hover_step_callback(None, source='decision')
            return

        mouse_point = self.plot.getViewBox().mapSceneToView(pos)
        x = mouse_point.x()
        if not self.dm.time_steps:
            return

        closest_idx = int(np.argmin(np.abs(np.arange(len(self.dm.time_steps)) - x)))
        if abs(closest_idx - x) > max(1, len(self.dm.time_steps) * 0.02):
            if self.hover_step_callback is not None:
                self.hover_step_callback(None, source='decision')
            return

        if self.hover_step_callback is not None:
            self.hover_step_callback(closest_idx, source='decision')

        time_step = self.dm.time_steps[closest_idx] if closest_idx < len(self.dm.time_steps) else closest_idx
        visible_curves = [curve for curve in self.curve_data if curve['curve'].isVisible()]
        if not visible_curves:
            return

        point_y = visible_curves[0]['values'][closest_idx]
        tooltip_html = (
            "<span style='color: #333; font-size: 9px;'>"
            f"<b>Step:</b> {time_step}<br>"
        )
        for curve in visible_curves:
            value = curve['values'][closest_idx]
            tooltip_html += (
                f"<span style='color: {curve['color_hex']};'>"
                f"<b>{curve['name']}:</b> {value:.2f}</span><br>"
            )
        tooltip_html += "</span>"

        self.hover_text = pg.TextItem(
            html=tooltip_html,
            anchor=(0, 0),
            border=pg.mkPen((50, 50, 50, 100), width=1),
            fill=pg.mkBrush(255, 255, 255, 235)
        )
        self.plot.addItem(self.hover_text, ignoreBounds=True)
        self._place_tooltip(self.hover_text, closest_idx, point_y)

        self.hover_line = pg.InfiniteLine(
            pos=closest_idx,
            angle=90,
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DotLine)
        )
        self.plot.addItem(self.hover_line, ignoreBounds=True)



class VisualizationWidget(QWidget):
    """Widget for displaying interactive visualizations of agent data"""
    
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager    # now available for decision plot
        # self.init_ui()

        # Setup the main window properties
        self.setWindowTitle("Crafter Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.frame_offset = 1 

        # Initialize the UI
        self.main_layout = QVBoxLayout(self)
        self.init_ui()
        
        # Initialize data containers
        self.time_steps = []
        self.reward_log = []
        self.action_log = []
        self.reward_components = {}
        self.cumulative_rewards = []  # Store pre-calculated cumulative rewards
        
        # Variable to track current position
        self.current_step = 0
        
        # Set initial view state
        self.view_state = {
            'cumulative': True,
            'components': True
        }
        
        # Initialize tracking variables for hover items
        self.component_curves = {}
        self.cumulative_data_points = []
        self.hover_text = None
        self.components_hover_text = None
        self.highlight_point = None
        self.hover_line = None
        self.component_display_order = []

    def _add_toggle_legend_row(self, layout, label_text, color_css, toggle_callback):
        row = LegendToggleRow(label_text, color_css, toggle_callback)
        layout.insertWidget(layout.count() - 1, row)
        return row
    
    def rebuild_decision_plot(self):
        # build only when there is at least one timestep
        if not self.data_manager.time_steps:
            return
        if hasattr(self, "decision_plot") and self.decision_plot is not None:
            self.decision_plot.setParent(None)
        self.decision_plot = DecisionAttribPlot(self.data_manager, hover_step_callback=self._sync_hover_step)

    
    def init_ui(self):
        """Initialize UI components for visualization"""
        
        # Set up the layout
        # layout = QVBoxLayout(self)
        # layout.setSpacing(4)  # Reduced spacing between plots
        
        # Create info panel at top with lower height
        self.info_panel = InfoPanel()
        # self.cumulative_plot = pg.PlotWidget(background="w")
        # self.cumulative_plot.setTitle("Cumulative Reward")
        # self.cumulative_plot.showGrid(x=True, y=True)
        # self.main_layout.addWidget(self.cumulative_plot, 6)  # Lower stretch factor
        
        # Create a plot widget for the cumulative reward with more space
        self.cumulative_plot = pg.PlotWidget(title="Agent Reward Timeline")
        self.cumulative_plot.setBackground('w')  # White background
        self.cumulative_plot.setLabel('left', 'Reward')
        self.cumulative_plot.setLabel('bottom', 'Time Step')
        self.cumulative_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cumulative_plot.setMouseEnabled(x=True, y=True)
        
        # Change text color to dark for better visibility
        self.cumulative_plot.getAxis('left').setTextPen('k')  # Black text
        self.cumulative_plot.getAxis('bottom').setTextPen('k')  # Black text
        self.cumulative_plot.setTitle("Agent Reward Timeline", color="#333", size="12pt")
        self.cumulative_plot.getViewBox().setDefaultPadding(0.0)
        self.cumulative_plot.getViewBox().setLimits(xMin=0)
        
        # Install event filter to clear hover items when mouse leaves plot
        self.cumulative_plot.installEventFilter(self)
        
        # Style improvements for cumulative plot
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        axis_pen = pg.mkPen(color=(120, 120, 120), width=1)
        self.cumulative_plot.getAxis('bottom').setPen(axis_pen)
        self.cumulative_plot.getAxis('left').setPen(axis_pen)
        self.cumulative_plot.getAxis('bottom').setStyle(tickFont=font)
        self.cumulative_plot.getAxis('left').setStyle(tickFont=font)
        
        # Wrap cumulative plot + external legend side-by-side.
        cum_container = QWidget()
        cum_row = QHBoxLayout(cum_container)
        cum_row.setContentsMargins(0, 0, 0, 0)
        cum_row.setSpacing(0)
        cum_row.addWidget(self.cumulative_plot, 1)

        self._cum_legend_inner = QWidget()
        self._cum_legend_inner.setStyleSheet("background: white; border-left: 1px solid #ddd;")
        self._cum_legend_vbox = QVBoxLayout(self._cum_legend_inner)
        self._cum_legend_vbox.setContentsMargins(8, 8, 8, 8)
        self._cum_legend_vbox.setSpacing(5)
        _cum_title = QLabel("Reward Timeline Legend")
        _cum_title.setStyleSheet("font-weight: bold; font-size: 10px; color: #333;")
        self._cum_legend_vbox.addWidget(_cum_title)
        self._cum_legend_vbox.addStretch(1)

        cum_legend_scroll = QScrollArea()
        cum_legend_scroll.setWidget(self._cum_legend_inner)
        cum_legend_scroll.setWidgetResizable(True)
        cum_legend_scroll.setFixedWidth(160)
        cum_legend_scroll.setFrameShape(QFrame.NoFrame)
        cum_legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cum_row.addWidget(cum_legend_scroll)

        # Set higher stretch factor for plot row
        self.main_layout.addWidget(cum_container, 6)

        # # Create the decision attribution comparison plot
        # self.decision_plot = DecisionAttribPlot(self.data_manager)
        # layout.addWidget(self.decision_plot, 4)  # reasonable space below cumulative

        
        # Create a plot for reward components with more space
        self.components_plot = pg.PlotWidget(title="Reward Component Breakdown")
        self.components_plot.setBackground('w')  # White background
        self.components_plot.setLabel('left', 'Component Value')
        self.components_plot.setLabel('bottom', 'Time Step')
        self.components_plot.showGrid(x=True, y=True, alpha=0.3)
        self.components_plot.setMouseEnabled(x=True, y=True)
        
        # Change text color to dark for better visibility
        self.components_plot.getAxis('left').setTextPen('k')  # Black text
        self.components_plot.getAxis('bottom').setTextPen('k')  # Black text
        self.components_plot.setTitle("Reward Component Breakdown", color="#333", size="12pt")
        self.components_plot.getViewBox().setDefaultPadding(0.0)
        self.components_plot.getViewBox().setLimits(xMin=0)
        
        # Install event filter to clear hover items when mouse leaves plot
        self.components_plot.installEventFilter(self)
        
        # Style improvements for components plot
        self.components_plot.getAxis('bottom').setPen(axis_pen)
        self.components_plot.getAxis('left').setPen(axis_pen)
        self.components_plot.getAxis('bottom').setStyle(tickFont=font)
        self.components_plot.getAxis('left').setStyle(tickFont=font)
        
        # Wrap components plot + external legend side-by-side in a container.
        comp_container = QWidget()
        comp_row = QHBoxLayout(comp_container)
        comp_row.setContentsMargins(0, 0, 0, 0)
        comp_row.setSpacing(0)
        comp_row.addWidget(self.components_plot, 1)

        # External legend panel — fixed width, scrollable, sits outside the plot.
        self._comp_legend_inner = QWidget()
        self._comp_legend_inner.setStyleSheet(
            "background: white; border-left: 1px solid #ddd;"
        )
        self._comp_legend_vbox = QVBoxLayout(self._comp_legend_inner)
        self._comp_legend_vbox.setContentsMargins(8, 8, 8, 8)
        self._comp_legend_vbox.setSpacing(5)
        _leg_title = QLabel("Components")
        _leg_title.setStyleSheet(
            "font-weight: bold; font-size: 10px; color: #333;"
        )
        self._comp_legend_vbox.addWidget(_leg_title)
        self._comp_legend_vbox.addStretch(1)

        legend_scroll = QScrollArea()
        legend_scroll.setWidget(self._comp_legend_inner)
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setFixedWidth(160)
        legend_scroll.setFrameShape(QFrame.NoFrame)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        comp_row.addWidget(legend_scroll)

        self.main_layout.addWidget(comp_container, 6)

        # Create position markers (vertical lines)
        self.cumulative_position_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen('r', width=2, style=Qt.DashLine)
        )
        self.components_position_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen('r', width=2, style=Qt.DashLine)
        )

        self.cumulative_plot.addItem(self.cumulative_position_line)
        self.components_plot.addItem(self.components_position_line)

        self.cumulative_sync_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(80, 80, 80), width=1, style=Qt.DotLine)
        )
        self.cumulative_sync_line.setVisible(False)
        self.cumulative_plot.addItem(self.cumulative_sync_line)

        self.components_sync_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(80, 80, 80), width=1, style=Qt.DotLine)
        )
        self.components_sync_line.setVisible(False)
        self.components_plot.addItem(self.components_sync_line)
        
        # Set up proxy for mouse hover events
        self.cumulative_proxy = pg.SignalProxy(
            self.cumulative_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.on_cumulative_hover
        )
        
        self.components_proxy = pg.SignalProxy(
            self.components_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.on_components_hover
        )
        
        # Add highlighted step region for context
        self.highlighted_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(100, 100, 255, 20),
            pen=pg.mkPen(None),
            movable=False
        )
        self.cumulative_plot.addItem(self.highlighted_region)
        self.highlighted_region.setVisible(False)
        
        # Add current step annotation
        self.current_step_text = pg.TextItem(
            text="",
            color=(200, 0, 0),
            anchor=(0.5, 0)
        )
        self.cumulative_plot.addItem(self.current_step_text)
        
        # Set plot size policies to allow expansion
        self.cumulative_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.components_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def eventFilter(self, obj, event):
        """Event filter to clear hover items when mouse leaves plot area"""
        if event.type() in (QEvent.Leave, QEvent.HoverLeave):
            if obj == self.cumulative_plot:
                self.clear_cumulative_hover_items()
                self._sync_hover_step(None, source='cumulative')
            elif obj == self.components_plot:
                self.clear_components_hover_items()
                self._sync_hover_step(None, source='components')
        return super().eventFilter(obj, event)

    def _sync_hover_step(self, step, source=None):
        """Synchronize hover timestep indicators across all three graphs."""
        if step is None or not self.time_steps:
            self.cumulative_sync_line.setVisible(False)
            self.components_sync_line.setVisible(False)
            if hasattr(self, 'decision_plot') and self.decision_plot is not None:
                self.decision_plot.clear_external_hover()
            return

        step = max(0, min(int(step), len(self.time_steps) - 1))
        x_pos = self.time_steps[step]

        if source != 'cumulative':
            self.cumulative_sync_line.setPos(x_pos)
            self.cumulative_sync_line.setVisible(True)
        else:
            self.cumulative_sync_line.setVisible(False)

        if source != 'components':
            self.components_sync_line.setPos(x_pos)
            self.components_sync_line.setVisible(True)
        else:
            self.components_sync_line.setVisible(False)

        if hasattr(self, 'decision_plot') and self.decision_plot is not None:
            if source != 'decision':
                self.decision_plot.set_external_hover_idx(step)
            else:
                self.decision_plot.clear_external_hover()
    
    def update_decision_marker(self, percent):
        if not self.data_manager.time_steps:
            return
        idx = int(percent * (len(self.data_manager.time_steps) - 1) / 100)
        self.decision_plot.update_marker(idx)

    
    def clear_cumulative_hover_items(self):
        """Clear all hover items from cumulative plot"""
        if hasattr(self, 'hover_text') and self.hover_text is not None:
            try:
                self.cumulative_plot.removeItem(self.hover_text)
            except Exception:
                pass
            self.hover_text = None
        
        if hasattr(self, 'highlight_point') and self.highlight_point is not None:
            try:
                self.cumulative_plot.removeItem(self.highlight_point)
            except Exception:
                pass
            self.highlight_point = None
    
    def clear_components_hover_items(self):
        """Clear all hover items from components plot"""
        if hasattr(self, 'components_hover_text') and self.components_hover_text is not None:
            try:
                self.components_plot.removeItem(self.components_hover_text)
            except Exception:
                pass
            self.components_hover_text = None
        
        if hasattr(self, 'hover_line') and self.hover_line is not None:
            try:
                self.components_plot.removeItem(self.hover_line)
            except Exception:
                pass
            self.hover_line = None

    def _place_plot_tooltip(self, plot_widget, text_item, point_x, point_y):
        vb = plot_widget.getViewBox()
        view_rect = plot_widget.viewRect()
        scene_rect = vb.sceneBoundingRect()
        data_per_px_x = view_rect.width() / max(scene_rect.width(), 1)
        data_per_px_y = view_rect.height() / max(scene_rect.height(), 1)
        text_rect = text_item.boundingRect()
        tip_w = text_rect.width() * data_per_px_x
        tip_h = text_rect.height() * data_per_px_y

        x_margin = data_per_px_x * 10
        y_margin = data_per_px_y * 8

        if point_x + x_margin + tip_w <= view_rect.right():
            anchor_x = 0
            x_pos = point_x + x_margin
        else:
            anchor_x = 1
            x_pos = point_x - x_margin

        if point_y + y_margin + tip_h <= view_rect.top():
            anchor_y = 0
            y_pos = point_y + y_margin
        elif point_y - y_margin - tip_h >= view_rect.bottom():
            anchor_y = 1
            y_pos = point_y - y_margin
        else:
            if (view_rect.top() - point_y) >= (point_y - view_rect.bottom()):
                anchor_y = 0
                y_pos = max(view_rect.bottom() + tip_h, min(point_y + y_margin, view_rect.top() - tip_h))
            else:
                anchor_y = 1
                y_pos = min(view_rect.top() - tip_h, max(point_y - y_margin, view_rect.bottom() + tip_h))

        text_item.setAnchor((anchor_x, anchor_y))
        text_item.setPos(x_pos, y_pos)

        item_rect = text_item.mapRectToParent(text_item.boundingRect())
        dx = 0
        dy = 0
        if item_rect.left() < view_rect.left():
            dx = view_rect.left() - item_rect.left()
        elif item_rect.right() > view_rect.right():
            dx = view_rect.right() - item_rect.right()
        if item_rect.bottom() < view_rect.bottom():
            dy = view_rect.bottom() - item_rect.bottom()
        elif item_rect.top() > view_rect.top():
            dy = view_rect.top() - item_rect.top()
        if dx or dy:
            text_item.setPos(text_item.pos().x() + dx, text_item.pos().y() + dy)
    
    def set_data(self, time_steps, reward_log, action_log, reward_components):
        """Set the data for visualization"""
        
        # Store the data
        self.time_steps = time_steps
        self.reward_log = reward_log
        self.action_log = action_log
        self.reward_components = reward_components
        
        # Pre-calculate cumulative rewards
        self.cumulative_rewards = np.cumsum(self.reward_log)
        
        # Clear existing plots
        self.cumulative_plot.clear()
        self.components_plot.clear()
        
        # Re-add position markers
        self.cumulative_plot.addItem(self.cumulative_position_line)
        self.components_plot.addItem(self.components_position_line)
        self.cumulative_plot.addItem(self.cumulative_sync_line)
        self.components_plot.addItem(self.components_sync_line)
        self.cumulative_plot.addItem(self.current_step_text)
        self.cumulative_plot.addItem(self.highlighted_region)
        
        # Clear timeline legend rows (preserve title + stretch).
        if hasattr(self, '_cum_legend_vbox'):
            # Keep: title (0), hint (1), stretch (last)
            while self._cum_legend_vbox.count() > 3:
                item = self._cum_legend_vbox.takeAt(2)
                if item.widget():
                    item.widget().deleteLater()

        # Clear the external legend panel (remove all label widgets except title + stretch).
        if hasattr(self, '_comp_legend_vbox'):
            # Keep: title (0), hint (1), stretch (last)
            while self._comp_legend_vbox.count() > 3:
                item = self._comp_legend_vbox.takeAt(2)
                if item.widget():
                    item.widget().deleteLater()
        
        # Update the plots with new data
        self.update_cumulative_plot()
        self.update_components_plot()
        
        # Initialize position at step 0
        self.update_position(0)
        
        # Store data points for hover lookup
        self.cumulative_data_points = []
        for i, (t, r, a, c) in enumerate(zip(self.time_steps, self.reward_log, 
                                           self.action_log, self.cumulative_rewards)):
            self.cumulative_data_points.append({
                'x': t,
                'y': c,
                'time_step': t,
                'action': a,
                'reward': r,
                'cumulative': c,
                'index': i
            })
        
        # Calculate visualization ranges and adjust axes
        if time_steps:
            x_min, x_max = min(time_steps), max(time_steps)
            
            # Find y-ranges with padding
            if self.cumulative_rewards.size > 0:
                y_min = min(0, np.min(self.cumulative_rewards))
                y_max = max(0, np.max(self.cumulative_rewards))
                y_padding = max((y_max - y_min) * 0.1, 0.5)  # 10% padding or at least 0.5
                
                # Set cumulative plot range
                self.cumulative_plot.setXRange(x_min, x_max, padding=0)
                self.cumulative_plot.setYRange(y_min - y_padding, y_max + y_padding)
            
            # Set component X range.
            self.components_plot.setXRange(x_min, x_max, padding=0)

            # Lock Y range from individual component values (non-stacked chart,
            # so per-component max IS the visible max).  Locking prevents hover
            # items from drifting the view even with ignoreBounds=False paths.
            if self.reward_components:
                all_vals = [v for vals in self.reward_components.values()
                            for v in vals if isinstance(v, (int, float))]
                if all_vals:
                    c_y_min = min(0.0, min(all_vals))
                    c_y_max = max(0.0, max(all_vals))
                    c_y_pad = max((c_y_max - c_y_min) * 0.12, 0.5)
                    self.components_plot.setYRange(
                        c_y_min - c_y_pad, c_y_max + c_y_pad, padding=0
                    )
            self.components_plot.getViewBox().disableAutoRange()
    
    def update_cumulative_plot(self):
        """Update the cumulative reward plot with interactive features"""
        
        if not self.time_steps or not self.reward_log:
            return
        
        # External legend entries for timeline plot.
        # Create main line plot
        pen = pg.mkPen(color=(0, 0, 255), width=2.5)
        self.cumulative_curve = self.cumulative_plot.plot(
            self.time_steps, 
            self.cumulative_rewards, 
            pen=pen, 
            name=None
        )
        
        positive_indices = [i for i, r in enumerate(self.reward_log) if r > 0.001]
        negative_indices = [i for i, r in enumerate(self.reward_log) if r < -0.001]

        self.positive_reward_points = None
        self.negative_reward_points = None
        self.positive_reward_bars = None
        self.negative_reward_bars = None

        if positive_indices:
            self.positive_reward_points = pg.ScatterPlotItem(
                [self.time_steps[i] for i in positive_indices],
                [self.cumulative_rewards[i] for i in positive_indices],
                symbol='t', size=14,
                pen=pg.mkPen(None), brush=pg.mkBrush(50, 200, 50, 220)
            )
            self.cumulative_plot.addItem(self.positive_reward_points)

        if negative_indices:
            self.negative_reward_points = pg.ScatterPlotItem(
                [self.time_steps[i] for i in negative_indices],
                [self.cumulative_rewards[i] for i in negative_indices],
                symbol='d', size=10,
                pen=pg.mkPen(None), brush=pg.mkBrush(200, 50, 50, 220)
            )
            self.cumulative_plot.addItem(self.negative_reward_points)
        
        # Add step rewards as a bar graph with custom styling
        if self.reward_log:
            try:
                positive_heights = [r if r > 0 else 0 for r in self.reward_log]
                negative_heights = [r if r < 0 else 0 for r in self.reward_log]

                if any(h > 0 for h in positive_heights):
                    self.positive_reward_bars = CustomBarGraphItem(
                        x=self.time_steps,
                        height=positive_heights,
                        width=0.6,
                        brushes=[pg.mkBrush(100, min(100 + h * 100, 255), 100, 150) for h in positive_heights],
                        pens=[pg.mkPen(0, 150, 0, 100, width=0.5) if h > 0 else pg.mkPen(None) for h in positive_heights]
                    )
                    self.cumulative_plot.addItem(self.positive_reward_bars)

                if any(h < 0 for h in negative_heights):
                    self.negative_reward_bars = CustomBarGraphItem(
                        x=self.time_steps,
                        height=negative_heights,
                        width=0.6,
                        brushes=[pg.mkBrush(min(100 + abs(h) * 100, 255), 100, 100, 150) for h in negative_heights],
                        pens=[pg.mkPen(150, 0, 0, 100, width=0.5) if h < 0 else pg.mkPen(None) for h in negative_heights]
                    )
                    self.cumulative_plot.addItem(self.negative_reward_bars)
            except Exception as e:
                print(f"Error creating custom bar graph: {e}")
            
        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, 
            angle=0, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
        )
        self.cumulative_plot.addItem(zero_line)

        if hasattr(self, '_cum_legend_vbox'):
            self._add_toggle_legend_row(
                self._cum_legend_vbox,
                "Positive Reward",
                "rgb(50,200,50)",
                lambda visible: [item.setVisible(visible) for item in [self.positive_reward_points, self.positive_reward_bars] if item is not None]
            )
            self._add_toggle_legend_row(
                self._cum_legend_vbox,
                "Negative Reward",
                "rgb(200,50,50)",
                lambda visible: [item.setVisible(visible) for item in [self.negative_reward_points, self.negative_reward_bars] if item is not None]
            )
            self._add_toggle_legend_row(
                self._cum_legend_vbox,
                "Cumulative Reward",
                "rgb(0,0,255)",
                lambda visible: self.cumulative_curve.setVisible(visible)
            )

    
    def update_components_plot(self):
        """Update the reward components plot - clean non-stacked overlapping area chart."""

        if not self.time_steps or not self.reward_components:
            return

        # Perceptually distinct palette; larger-amplitude components drawn first
        # (background) so smaller ones remain readable on top.
        colors = [
            (220,  55,  55),   # Red
            ( 30, 160,  90),   # Green
            ( 50, 115, 240),   # Blue
            (240, 165,   0),   # Amber
            (170,   0, 210),   # Purple
            (  0, 185, 185),   # Cyan
            (240, 110,   0),   # Orange
            (130,  50, 230),   # Violet
            (  0, 140, 120),   # Teal
            (190,  30,  90),   # Crimson
        ]

        self.component_curves = {}
        self.component_display_order = []

        # Only keep components that actually contain non-zero values.
        active = {k: v for k, v in self.reward_components.items()
                  if any(val != 0 for val in v)}
        if not active:
            return

        # Draw largest-amplitude component first so it sits in the background.
        sorted_components = sorted(
            active.items(),
            key=lambda kv: max(abs(min(kv[1])), abs(max(kv[1]))),
            reverse=True,
        )

        x = np.array(self.time_steps)

        for idx, (name, values) in enumerate(sorted_components):
            r, g, b = colors[idx % len(colors)]
            y = np.array(values, dtype=float)
            self.component_display_order.append(name)

            # Each component fills from y=0 to its own value (non-stacked).
            # fillLevel=0 handles both positive and negative values correctly.
            curve = self.components_plot.plot(
                x, y,
                pen=pg.mkPen(color=(r, g, b), width=2),
                fillLevel=0.0,
                brush=pg.mkBrush(r, g, b, 45),
                name=name,
            )

            self.component_curves[name] = {
                'curve': curve,
                'values': list(values),
                'color': (r, g, b),
            }

            # Add a row to the external legend panel.
            if hasattr(self, '_comp_legend_vbox'):
                self._add_toggle_legend_row(
                    self._comp_legend_vbox,
                    name.replace('_', ' '),
                    f"rgb({r},{g},{b})",
                    lambda visible, item=curve: item.setVisible(visible)
                )

        # Subtle zero reference line.
        self.components_plot.addItem(
            pg.InfiniteLine(
                pos=0, angle=0,
                pen=pg.mkPen(color=(80, 80, 80), width=1, style=Qt.DashLine)
            )
        )
    
    def update_position(self, step, from_video=False, from_timeline=False):
        """Update the position marker in visualizations"""
        
        if not self.time_steps or step >= len(self.time_steps):
            print(f"Invalid step: {step}, total steps: {len(self.time_steps)}")
            return
        
        # Update current step
        self.current_step = step
        
        # Get the x-position (time step)
        x_pos = self.time_steps[step]
        
        # Update position lines
        self.cumulative_position_line.setValue(x_pos)
        self.components_position_line.setValue(x_pos)
        if hasattr(self, 'decision_plot') and self.decision_plot is not None:
            self.decision_plot.update_marker(step)
        
        # Update highlighted region for context (5 steps before and after)
        context_start = max(0, step - 5)
        context_end = min(len(self.time_steps) - 1, step + 5)
        if context_start < context_end:
            self.highlighted_region.setRegion([
                self.time_steps[context_start], 
                self.time_steps[context_end]
            ])
            self.highlighted_region.setVisible(True)
        else:
            self.highlighted_region.setVisible(False)
        
        # Update step annotation (without Step/Action/Reward labels)
        if self.action_log and step < len(self.action_log):
            action = self.action_log[step]
            
            # Create minimal marker for the current position
            self.current_step_text.setHtml(
                f"<div style='background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; "
                f"border: 1px solid #aaa; font-size: 9px;'>"
                f"<span style='color: #000;'>{x_pos}</span>"
                f"</div>"
            )
            self.current_step_text.setPos(x_pos, self.cumulative_rewards[step])
            
            # Update info panel with detailed state information
            if hasattr(self, 'info_panel'):
                # Gather all data for this step
                step_data = {
                    'time_step': x_pos,
                    'action': action,
                    'reward': self.reward_log[step],
                    'cumulative_reward': self.cumulative_rewards[step]
                }
                
                # Add component values
                for name, values in self.reward_components.items():
                    if step < len(values):
                        step_data[name] = values[step]
                
                self.info_panel.update_state(step_data)
    
    def on_cumulative_hover(self, event):
        """Handle mouse hover over cumulative plot with adaptive tooltip placement."""
        self.clear_cumulative_hover_items()
        
        if not event or not self.time_steps or not self.reward_log:
            return
        
        # Map mouse position to data coordinates
        pos = event[0]
        if not self.cumulative_plot.sceneBoundingRect().contains(pos):
            self._sync_hover_step(None, source='cumulative')
            return
        plot_item = self.cumulative_plot.getPlotItem()
        view_box = plot_item.getViewBox()
        mouse_point = view_box.mapSceneToView(pos)
        
        # Find closest point to mouse
        if len(self.time_steps) == 0:
            return
            
        # Find closest x coordinate
        x_coord = mouse_point.x()
        closest_idx = np.argmin(np.abs(np.array(self.time_steps) - x_coord))
        
        if closest_idx >= len(self.time_steps):
            return
            
        closest_time = self.time_steps[closest_idx]
        closest_reward = self.cumulative_rewards[closest_idx]
        
        # If mouse is too far from point, don't show tooltip
        if abs(closest_time - x_coord) > (max(self.time_steps) - min(self.time_steps)) * 0.02:
            self._sync_hover_step(None, source='cumulative')
            return
        
        # Get action name
        action_value = self.action_log[closest_idx]
        action_name = "Unknown"
        if self.data_manager:
            action_name = self.data_manager.get_action_name(action_value)
        
        tooltip_html = (
            "<span style='color: #333; font-size: 9px;'>"
            f"<b>Step:</b> {closest_time}<br>"
            f"<b>Action:</b> {action_name} ({action_value})<br>"
            f"<b>Reward:</b> {self.reward_log[closest_idx]:.2f}<br>"
            f"<b>Cumulative:</b> {closest_reward:.2f}</span>"
        )

        self.hover_text = pg.TextItem(
            html=tooltip_html,
            anchor=(0, 0),
            border=pg.mkPen((50, 50, 50, 100), width=1),
            fill=pg.mkBrush(255, 255, 255, 235)
        )
        self.cumulative_plot.addItem(self.hover_text, ignoreBounds=True)

        self._place_plot_tooltip(self.cumulative_plot, self.hover_text, closest_time, closest_reward)
        
        # Create highlight point
        self.highlight_point = pg.ScatterPlotItem(
            [closest_time], [closest_reward], 
            size=12, pen=(200, 200, 200), brush=(255, 255, 0, 200),
            symbol='o'
        )
        self.cumulative_plot.addItem(self.highlight_point, ignoreBounds=True)
        self._sync_hover_step(closest_idx, source='cumulative')

    
    def on_components_hover(self, event):
        """Handle hover events on the components plot with stable, adaptive tooltips."""

        self.clear_components_hover_items()

        if not hasattr(self, 'component_curves') or not self.component_curves:
            self._sync_hover_step(None, source='components')
            return

        pos = event[0]
        if not self.components_plot.sceneBoundingRect().contains(pos):
            self._sync_hover_step(None, source='components')
            return

        mouse_point = self.components_plot.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        if not self.time_steps:
            self._sync_hover_step(None, source='components')
            return

        closest_idx = min(range(len(self.time_steps)), key=lambda i: abs(self.time_steps[i] - x))

        # Keep tooltips only when mouse is reasonably near a step.
        if abs(self.time_steps[closest_idx] - x) > (max(self.time_steps) - min(self.time_steps)) / 20:
            self._sync_hover_step(None, source='components')
            return

        # Gather component values at hovered step.
        component_values = {}
        for name, data in self.component_curves.items():
            values = data['values']
            if closest_idx < len(values):
                component_values[name] = values[closest_idx]

        # Resolve action name (matching cumulative tooltip behavior).
        action_value = self.action_log[closest_idx] if closest_idx < len(self.action_log) else None
        action_name = "Unknown"
        if action_value is not None and self.data_manager:
            action_name = self.data_manager.get_action_name(action_value)

        shown_components = []
        for name in self.component_display_order:
            value = component_values.get(name)
            if value is not None and value > 0:
                shown_components.append((name, value))

        tooltip_html = (
            "<span style='color: #333; font-size: 9px;'>"
            f"<b>Step:</b> {self.time_steps[closest_idx]}<br>"
            f"<b>Action:</b> {action_name} ({action_value})<br>"
        )

        for name, value in shown_components:
            color = self.component_curves[name]['color']
            tooltip_html += (
                f"<span style='color: rgb({color[0]},{color[1]},{color[2]});'>"
                f"<b>{name}:</b> {value:.2f}</span><br>"
            )

        if not shown_components:
            tooltip_html += "<span style='color: #666;'>No positive components</span><br>"

        tooltip_html += "</span>"

        self.components_hover_text = pg.TextItem(
            html=tooltip_html,
            anchor=(0, 0),
            border=pg.mkPen((50, 50, 50, 100), width=1),
            fill=pg.mkBrush(255, 255, 255, 235)
        )

        # Add first so boundingRect is measurable for data-space clamping.
        self.components_plot.addItem(self.components_hover_text, ignoreBounds=True)

        point_x = self.time_steps[closest_idx]
        point_y = y
        self._place_plot_tooltip(self.components_plot, self.components_hover_text, point_x, point_y)

        self.hover_line = pg.InfiniteLine(
            pos=point_x,
            angle=90,
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DotLine)
        )
        self.components_plot.addItem(self.hover_line, ignoreBounds=True)
        self._sync_hover_step(closest_idx, source='components')
    
    def toggle_view(self, view_type, visible):
        """Toggle visibility of different visualization types"""
        
        self.view_state[view_type] = visible
        
        # Update visibility
        if view_type == 'cumulative':
            self.cumulative_plot.setVisible(visible)
        elif view_type == 'components':
            self.components_plot.setVisible(visible)
        elif view_type == 'decision':
            self.decision_plot.setVisible(visible)
