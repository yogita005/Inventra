import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import requests
from sklearn.cluster import KMeans
import joblib
import base64
from pathlib import Path

class AdvancedInventoryManagementTwin:
    def __init__(self, inventory_file="inventory_data.json", model_file="inventory_model.joblib"):
        self.inventory_file = inventory_file
        self.model_file = model_file
        self.history_file = "inventory_history.json"
        self.settings_file = "system_settings.json"
        
        # Define inventory items with color ranges in HSV
        self.known_items = {
            "apple": {
                "count": 0, 
                "color": [0, 0, 255],  # BGR Red
                "min_hsv": np.array([0, 100, 100]), 
                "max_hsv": np.array([10, 255, 255]),
                "min_stock": 5,
                "unit_price": 0.75,
                "supplier": "FreshFruits Inc.",
                "category": "fruit",
                "shelf_life_days": 14
            },
            "banana": {
                "count": 0, 
                "color": [0, 255, 255],  # BGR Yellow
                "min_hsv": np.array([20, 100, 100]), 
                "max_hsv": np.array([35, 255, 255]),
                "min_stock": 8,
                "unit_price": 0.25,
                "supplier": "TropicalGoods",
                "category": "fruit",
                "shelf_life_days": 7
            },
            "orange": {
                "count": 0, 
                "color": [0, 165, 255],  # BGR Orange
                "min_hsv": np.array([10, 100, 100]), 
                "max_hsv": np.array([20, 255, 255]),
                "min_stock": 6,
                "unit_price": 0.80,
                "supplier": "CitrusSuppliers",
                "category": "fruit",
                "shelf_life_days": 21
            },
            "blue_item": {
                "count": 0, 
                "color": [255, 0, 0],  # BGR Blue
                "min_hsv": np.array([100, 100, 100]), 
                "max_hsv": np.array([130, 255, 255]),
                "min_stock": 3,
                "unit_price": 1.50,
                "supplier": "BlueProducts Ltd.",
                "category": "manufactured",
                "shelf_life_days": 365
            },
            "green_item": {
                "count": 0, 
                "color": [0, 255, 0],  # BGR Green
                "min_hsv": np.array([35, 100, 100]), 
                "max_hsv": np.array([85, 255, 255]),
                "min_stock": 4,
                "unit_price": 1.25,
                "supplier": "GreenGoods Co.",
                "category": "manufactured",
                "shelf_life_days": 365
            }
        }
        
        # System settings
        self.settings = {
            "min_object_area": 500,
            "detection_confidence_threshold": 0.60,
            "auto_reorder": False,
            "notify_low_stock": True,
            "display_mode": "standard",
            "advanced_features": True,
            "retention_period_days": 90
        }
        
        # Load settings and inventory
        self.load_settings()
        self.load_inventory()
        self.load_inventory_history()
        
        # Initialize ML model
        self.prediction_model = None
        self.load_or_create_model()
        
    def load_settings(self):
        """Load system settings if available"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
            except Exception as e:
                st.error(f"Error loading settings: {e}")
                
    def save_settings(self):
        """Save current system settings"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=4)
        
    def load_inventory(self):
        """Load existing inventory data if available"""
        if os.path.exists(self.inventory_file):
            try:
                with open(self.inventory_file, 'r') as f:
                    saved_data = json.load(f)
                    for item, data in saved_data.items():
                        if item in self.known_items:
                            # Update only keys that exist in saved data
                            for key, value in data.items():
                                self.known_items[item][key] = value
            except Exception as e:
                st.error(f"Error loading inventory data: {e}")
        else:
            # Initialize with timestamps
            for item in self.known_items:
                self.known_items[item]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.known_items[item]["expiry_date"] = (datetime.now() + 
                                                        timedelta(days=self.known_items[item]["shelf_life_days"])
                                                        ).strftime("%Y-%m-%d")
    
    def load_inventory_history(self):
        """Load historical inventory data"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.inventory_history = json.load(f)
            except Exception as e:
                st.error(f"Error loading inventory history: {e}")
                self.inventory_history = {"timestamps": [], "data": {}}
        else:
            # Initialize empty history
            self.inventory_history = {"timestamps": [], "data": {}}
            for item in self.known_items:
                self.inventory_history["data"][item] = []
    
    def update_inventory_history(self):
        """Add current inventory state to history"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add timestamp if it doesn't exist yet
        if current_time not in self.inventory_history["timestamps"]:
            self.inventory_history["timestamps"].append(current_time)
            
            # Add count for each item
            for item, data in self.known_items.items():
                self.inventory_history["data"][item].append(data["count"])
                
            # Keep only data within retention period
            retention_days = self.settings["retention_period_days"]
            cutoff_date = (datetime.now() - timedelta(days=retention_days))
            
            # Filter history by date
            valid_timestamps = []
            valid_indices = []
            
            for i, timestamp in enumerate(self.inventory_history["timestamps"]):
                ts_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                if ts_date >= cutoff_date:
                    valid_timestamps.append(timestamp)
                    valid_indices.append(i)
            
            # Update timestamps and data
            self.inventory_history["timestamps"] = valid_timestamps
            for item in self.inventory_history["data"]:
                self.inventory_history["data"][item] = [self.inventory_history["data"][item][i] for i in valid_indices]
            
            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(self.inventory_history, f, indent=4)
    
    def save_inventory(self):
        """Save current inventory data to file"""
        save_data = {}
        for item, data in self.known_items.items():
            save_data[item] = {k: v for k, v in data.items() if k != "color" and k != "min_hsv" and k != "max_hsv"}
        
        with open(self.inventory_file, 'w') as f:
            json.dump(save_data, f, indent=4)
            
        # Update inventory history
        self.update_inventory_history()
    
    def load_or_create_model(self):
        """Load existing prediction model or create a new one"""
        if os.path.exists(self.model_file):
            try:
                self.prediction_model = joblib.load(self.model_file)
            except Exception as e:
                st.warning(f"Could not load prediction model: {e}")
                self.prediction_model = None
        else:
            self.prediction_model = None
    
    def train_prediction_model(self):
        """Train a machine learning model to predict future inventory needs"""
        if len(self.inventory_history["timestamps"]) < 10:
            return False, "Not enough historical data for training (need at least 10 data points)"
        
        try:
            # Prepare training data for each item
            all_features = []
            all_targets = []
            window_size = 3
            
            for item in self.known_items:
                item_data = self.inventory_history["data"][item]
                
                if len(item_data) <= window_size:
                    continue
                    
                for i in range(len(item_data) - window_size):
                    # Features: window_size previous values
                    features = item_data[i:i+window_size]
                    # Target: next value
                    target = item_data[i+window_size]
                    
                    all_features.append(features)
                    all_targets.append(target)
            
            if len(all_features) == 0:
                return False, "Not enough sequential data for training"
            model = KMeans(n_clusters=5, random_state=42)
            model.fit(all_features)
            
            self.prediction_model = model
            joblib.dump(model, self.model_file)
            
            return True, "Model trained successfully"
            
        except Exception as e:
            return False, f"Error training model: {e}"
    
    def predict_future_inventory(self, item, days_ahead=7):
        """Predict future inventory levels"""
        if self.prediction_model is None or item not in self.known_items:
            return None
            
        if len(self.inventory_history["data"][item]) < 3:
            return None
            
        # Get last 3 inventory counts for this item
        recent_counts = self.inventory_history["data"][item][-3:]
        
        if len(recent_counts) < 3:
            return None
            
        # Use model to predict future pattern (simplified)
        cluster = self.prediction_model.predict([recent_counts])[0]
        
        # Basic linear extrapolation based on recent trend
        if len(recent_counts) >= 2:
            avg_daily_change = (recent_counts[-1] - recent_counts[0]) / len(recent_counts)
            predicted_count = max(0, recent_counts[-1] + (avg_daily_change * days_ahead))
            return round(predicted_count, 1)
        
        return None
    
    def get_items_to_reorder(self):
        """Get list of items that need to be reordered"""
        reorder_items = []
        
        for item, data in self.known_items.items():
            current_count = data["count"]
            min_stock = data.get("min_stock", 3)
            
            # Check if current stock is below minimum threshold
            if current_count < min_stock:
                reorder_amount = (min_stock * 2) - current_count  # Reorder to twice the minimum
                
                reorder_items.append({
                    "item": item.replace("_", " ").capitalize(),
                    "current": current_count,
                    "minimum": min_stock,
                    "to_order": reorder_amount,
                    "supplier": data.get("supplier", "Unknown"),
                    "cost": round(reorder_amount * data.get("unit_price", 1.0), 2)
                })
                
        return reorder_items
    
    def auto_generate_purchase_order(self):
        """Generate a purchase order for items that need reordering"""
        items_to_reorder = self.get_items_to_reorder()
        
        if not items_to_reorder:
            return "No items need reordering at this time."
            
        # Group items by supplier
        suppliers = {}
        for item in items_to_reorder:
            supplier = item["supplier"]
            if supplier not in suppliers:
                suppliers[supplier] = []
            suppliers[supplier].append(item)
            
        # Generate PO text
        po_text = "PURCHASE ORDER\n"
        po_text += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        po_text += f"PO Number: PO-{datetime.now().strftime('%Y%m%d%H%M')}\n\n"
        
        total_cost = 0
        
        for supplier, items in suppliers.items():
            po_text += f"Supplier: {supplier}\n"
            po_text += "-" * 50 + "\n"
            po_text += "Item                  Quantity    Unit Price    Total\n"
            po_text += "-" * 50 + "\n"
            
            supplier_total = 0
            
            for item in items:
                item_name = item["item"]
                quantity = item["to_order"]
                unit_price = self.known_items[item_name.lower().replace(" ", "_")].get("unit_price", 0)
                item_total = quantity * unit_price
                
                po_text += f"{item_name.ljust(20)} {str(quantity).ljust(10)} ${unit_price:.2f}         ${item_total:.2f}\n"
                supplier_total += item_total
                
            po_text += "-" * 50 + "\n"
            po_text += f"Supplier Total: ${supplier_total:.2f}\n\n"
            total_cost += supplier_total
            
        po_text += "=" * 50 + "\n"
        po_text += f"TOTAL ORDER COST: ${total_cost:.2f}\n"
        
        return po_text
        
    def detect_objects(self, frame):
        """Detect objects using improved color thresholding with advanced features"""
        # Convert to HSV for better color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result_frame = frame.copy()
        detected_items = {item: 0 for item in self.known_items}
        
        # Apply Gaussian blur to reduce noise
        blurred_hsv = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
        
        # Process each known item type
        for item_name, item_data in self.known_items.items():
            # Create mask for the specific color range
            mask = cv2.inRange(blurred_hsv, item_data["min_hsv"], item_data["max_hsv"])
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.settings["min_object_area"]:
                    # Calculate contour features for better detection
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
                    area_factor = min(area / 10000, 0.8)
                    
                    # Shape factor
                    shape_factor = min(circularity, 0.8) if item_data.get("category") == "fruit" else 0.5
                    
                    # Calculate color consistency within the contour
                    mask_contour = np.zeros_like(mask)
                    cv2.drawContours(mask_contour, [contour], 0, 255, -1)
                    roi_hsv = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask_contour)
                    
                    # Get non-zero pixels (i.e., the object)
                    nonzero_pixels = roi_hsv[np.where(mask_contour > 0)]
                    
                    # Calculate color consistency if we have pixels
                    color_factor = 0.5
                    if nonzero_pixels.size > 0:
                        # Calculate standard deviation of hue
                        h_values = nonzero_pixels[:, 0]
                        h_std = np.std(h_values)
                        # Lower standard deviation means more consistent color
                        color_factor = min(0.8, max(0.3, 1.0 - (h_std / 30)))
                    
                    # Combined confidence score (weighted average)
                    confidence = (0.4 * area_factor + 0.3 * shape_factor + 0.3 * color_factor)
                    
                    # Apply threshold from settings
                    if confidence >= self.settings["detection_confidence_threshold"]:
                        detected_items[item_name] += 1
                        
                        # Draw bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(result_frame, (x, y), (x + w, y + h), item_data["color"], 2)
                        
                        # Improved contour visualization
                        cv2.drawContours(result_frame, [contour], 0, item_data["color"], 2)
                        
                        # Add label with confidence
                        label = f"{item_name.replace('_', ' ').capitalize()}: {confidence:.2f}"
                        cv2.putText(result_frame, label, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, item_data["color"], 2)
        
        return result_frame, detected_items
    
    def process_frame(self, frame):
        """Process a video frame to detect and count items"""
        # Resize image if too large (improves performance)
        height, width = frame.shape[:2]
        max_dimension = 800
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect objects
        result_frame, detected_items = self.detect_objects(frame)
        
        # Add image augmentations based on display mode
        if self.settings["display_mode"] == "enhanced":
            # Add grid overlay
            h, w = result_frame.shape[:2]
            grid_size = 50
            for x in range(0, w, grid_size):
                cv2.line(result_frame, (x, 0), (x, h), (50, 50, 50), 1)
            for y in range(0, h, grid_size):
                cv2.line(result_frame, (0, y), (w, y), (50, 50, 50), 1)
                
            # Add detection statistics
            total_detected = sum(detected_items.values())
            cv2.putText(result_frame, f"Objects: {total_detected}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update inventory counts if they've changed
        for item, count in detected_items.items():
            if count != self.known_items[item]["count"]:
                self.known_items[item]["count"] = count
                self.known_items[item]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Update expiry date for perishable items
                if self.known_items[item].get("category") == "fruit":
                    self.known_items[item]["expiry_date"] = (
                        datetime.now() + timedelta(days=self.known_items[item]["shelf_life_days"])
                    ).strftime("%Y-%m-%d")
        
        # Save updated inventory
        self.save_inventory()
        
        return result_frame, detected_items
        
    def get_expiring_items(self, days_threshold=3):
        """Get items that will expire soon"""
        expiring_items = []
        today = datetime.now().date()
        
        for item, data in self.known_items.items():
            if data["count"] > 0 and "expiry_date" in data and data.get("category") == "fruit":
                try:
                    expiry_date = datetime.strptime(data["expiry_date"], "%Y-%m-%d").date()
                    days_until_expiry = (expiry_date - today).days
                    
                    if 0 <= days_until_expiry <= days_threshold:
                        expiring_items.append({
                            "item": item.replace("_", " ").capitalize(),
                            "count": data["count"],
                            "expiry_date": data["expiry_date"],
                            "days_left": days_until_expiry
                        })
                except Exception:
                    continue
                    
        return expiring_items
        
    def get_inventory_value(self):
        """Calculate total inventory value"""
        total_value = sum(data["count"] * data.get("unit_price", 0) 
                          for item, data in self.known_items.items())
        return round(total_value, 2)
        
    def generate_inventory_report(self):
        """Generate a comprehensive inventory report"""
        # Basic inventory status
        report = "# Inventory Status Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary section
        report += "## Summary\n\n"
        total_items = sum(data["count"] for data in self.known_items.values())
        total_value = self.get_inventory_value()
        unique_items = sum(1 for data in self.known_items.values() if data["count"] > 0)
        
        report += f"* **Total Items:** {total_items}\n"
        report += f"* **Unique Items:** {unique_items}\n"
        report += f"* **Total Value:** ${total_value}\n\n"
        
        # Items section
        report += "## Inventory Items\n\n"
        report += "| Item | Quantity | Value | Last Updated | Status |\n"
        report += "| ---- | -------- | ----- | ------------ | ------ |\n"
        
        for item, data in self.known_items.items():
            item_name = item.replace("_", " ").capitalize()
            quantity = data["count"]
            value = round(quantity * data.get("unit_price", 0), 2)
            last_updated = data.get("last_updated", "N/A")
            
            # Determine status
            if quantity <= 0:
                status = "⚠️ Out of Stock"
            elif quantity < data.get("min_stock", 3):
                status = "⚠️ Low Stock"
            else:
                status = "✅ In Stock"
                
            report += f"| {item_name} | {quantity} | ${value} | {last_updated} | {status} |\n"
            
        # Alerts section
        report += "\n## Alerts\n\n"
        
        # Low stock alerts
        low_stock = [(item.replace("_", " ").capitalize(), data["count"], data.get("min_stock", 3)) 
                     for item, data in self.known_items.items() 
                     if 0 < data["count"] < data.get("min_stock", 3)]
        
        if low_stock:
            report += "### Low Stock Alerts\n\n"
            for item, count, min_stock in low_stock:
                report += f"* {item}: {count} (minimum: {min_stock})\n"
        else:
            report += "No low stock alerts.\n"
            
        # Expiring items
        expiring = self.get_expiring_items(7)
        if expiring:
            report += "\n### Expiration Alerts\n\n"
            for item in expiring:
                report += f"* {item['item']}: {item['count']} units expire in {item['days_left']} days ({item['expiry_date']})\n"
        
        # Recommendations section
        report += "\n## Recommendations\n\n"
        
        # Reorder recommendations
        reorder_items = self.get_items_to_reorder()
        if reorder_items:
            report += "### Recommended Reorders\n\n"
            for item in reorder_items:
                report += f"* Order {item['to_order']} units of {item['item']} from {item['supplier']} (${item['cost']})\n"
        else:
            report += "No reorders needed at this time.\n"
            
        # Future predictions if model exists
        if self.prediction_model is not None:
            report += "\n### 7-Day Predictions\n\n"
            report += "| Item | Current | Predicted | Change |\n"
            report += "| ---- | ------- | --------- | ------ |\n"
            
            for item in self.known_items:
                current = self.known_items[item]["count"]
                predicted = self.predict_future_inventory(item, 7)
                
                if predicted is not None:
                    change = predicted - current
                    change_str = f"+{change}" if change > 0 else f"{change}"
                    item_name = item.replace("_", " ").capitalize()
                    report += f"| {item_name} | {current} | {predicted} | {change_str} |\n"
        
        return report

def get_sample_images():
    """Returns a dictionary of sample images for testing"""
    sample_images = {
        "Fruits on table": "https://raw.githubusercontent.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/master/Intermediate/Custom%20Object%20Detection/Resources/apple.jpg",
        "Colorful objects": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/colored_toys.jpg", 
        "Grocery items": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg"
    }
    return sample_images

def load_image_from_url(url):
    """Load an image from a URL"""
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate HTML code for file download link"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def app():
    st.set_page_config(
        page_title="Inventory Digital Twin",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📦 Inventory Management Digital Twin")
    st.markdown("An AI-powered computer vision system for tracking inventory with predictive analytics")
    
    # Initialize inventory system with improved error handling
    try:
        inventory_system = AdvancedInventoryManagementTwin()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()
    
    # Sidebar settings
    st.sidebar.header("Configuration")
    
    # Advanced settings section
    with st.sidebar.expander("Detection Settings", expanded=False):
        # Adjust color sensitivity
        color_sensitivity = st.slider("Color Sensitivity", 10, 100, 50)
        
        # Update color ranges based on sensitivity
        for item in inventory_system.known_items:
            # Adjust the saturation and value ranges based on sensitivity
            base_min = inventory_system.known_items[item]["min_hsv"].copy()
            base_max = inventory_system.known_items[item]["max_hsv"].copy()
            
            # Adjust saturation range (higher sensitivity = lower minimum saturation)
            sat_adjustment = 100 - color_sensitivity
            inventory_system.known_items[item]["min_hsv"][1] = max(50, base_min[1] - sat_adjustment)
        
        # Adjust minimum object area
        min_object_size = st.slider("Minimum Object Size", 100, 2000, 
                                   inventory_system.settings["min_object_area"])
        inventory_system.settings["min_object_area"] = min_object_size
        
        # Confidence threshold
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.3, 1.0, 
                                        inventory_system.settings["detection_confidence_threshold"], 
                                        step=0.05)
        inventory_system.settings["detection_confidence_threshold"] = confidence_threshold
        
        # Display mode
        display_mode = st.selectbox("Display Mode", 
                                   ["standard", "enhanced"],
                                   index=0 if inventory_system.settings["display_mode"] == "standard" else 1)
        inventory_system.settings["display_mode"] = display_mode
    
    # System features
    with st.sidebar.expander("System Features", expanded=False):
        auto_reorder = st.checkbox("Enable Auto-Reorder Alerts", 
                                 inventory_system.settings["auto_reorder"])
        
        notify_low_stock = st.checkbox("Enable Low Stock Notifications", 
                                    inventory_system.settings["notify_low_stock"])
        inventory_system.settings["notify_low_stock"] = notify_low_stock
        
        enable_advanced = st.checkbox("Enable Advanced Analytics", 
                                    inventory_system.settings["advanced_features"])
        inventory_system.settings["advanced_features"] = enable_advanced
        
        retention_period = st.slider("Data Retention Period (days)", 
                                   30, 365, 
                                   inventory_system.settings["retention_period_days"])
        inventory_system.settings["retention_period_days"] = retention_period
        
        # Save settings button
        if st.button("Save Settings"):
            inventory_system.save_settings()
            st.success("Settings saved successfully!")
    
    # Main area tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Inventory Detection", "Dashboard", "Reports", "ML Predictions"])
    
    with tab1:
        st.header("Inventory Object Detection")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Image input options
            image_source = st.radio("Select Image Source", 
                                   ["Upload Image", "Sample Images", "Camera"])
            
            if image_source == "Sample Images":
                sample_images = get_sample_images()
                selected_sample = st.selectbox("Choose a sample image", 
                                             list(sample_images.keys()))
                selected_image_url = sample_images[selected_sample]
                
                if st.button("Process Sample Image"):
                    with st.spinner("Processing image..."):
                        # Load and process sample image
                        img = load_image_from_url(selected_image_url)
                        if img is not None:
                            result_img, detected = inventory_system.process_frame(img)
                            
                            # Convert BGR to RGB for display
                            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            with col1:
                                st.image(result_rgb, caption="Processed Image", use_column_width=True)
                                
                                # Show detection results
                                st.write("### Detection Results")
                                results_df = pd.DataFrame({
                                    "Item": [item.replace("_", " ").capitalize() for item in detected.keys()],
                                    "Count": list(detected.values())
                                })
                                st.dataframe(results_df)
            
            elif image_source == "Upload Image":
                uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    if st.button("Process Uploaded Image"):
                        with st.spinner("Processing image..."):
                            # Read uploaded image
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            
                            result_img, detected = inventory_system.process_frame(img)
                            
                            # Convert BGR to RGB for display
                            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            with col1:
                                st.image(result_rgb, caption="Processed Image", use_column_width=True)
                                
                                # Show detection results
                                st.write("### Detection Results")
                                results_df = pd.DataFrame({
                                    "Item": [item.replace("_", " ").capitalize() for item in detected.keys()],
                                    "Count": list(detected.values())
                                })
                                st.dataframe(results_df)
            
            elif image_source == "Camera":
                st.write("Camera capture")
                camera_image = st.camera_input("Take a picture")
                
                if camera_image is not None and st.button("Process Camera Image"):
                    with st.spinner("Processing image..."):
                        # Read camera image
                        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                        result_img, detected = inventory_system.process_frame(img)
                        
                        # Convert BGR to RGB for display
                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        with col1:
                            st.image(result_rgb, caption="Processed Image", use_column_width=True)
                            
                            # Show detection results
                            st.write("### Detection Results")
                            results_df = pd.DataFrame({
                                "Item": [item.replace("_", " ").capitalize() for item in detected.keys()],
                                "Count": list(detected.values())
                            })
                            st.dataframe(results_df)
        
        # Initial content for col1 if no image is processed yet
        with col1:
            if "result_rgb" not in locals():
                st.info("Select an image source and click 'Process' to detect inventory items.")
                
    with tab2:
        st.header("Inventory Dashboard")
        
        # Current inventory summary
        st.subheader("Current Inventory")
        
        # Create table of current inventory
        inventory_data = []
        for item, data in inventory_system.known_items.items():
            status = "In Stock"
            if data["count"] <= 0:
                status = "Out of Stock"
            elif data["count"] < data.get("min_stock", 3):
                status = "Low Stock"
                
            inventory_data.append({
                "Item": item.replace("_", " ").capitalize(),
                "Count": data["count"],
                "Minimum Stock": data.get("min_stock", 3),
                "Unit Price": f"${data.get('unit_price', 0):.2f}",
                "Value": f"${data['count'] * data.get('unit_price', 0):.2f}",
                "Status": status,
                "Last Updated": data.get("last_updated", "N/A")
            })
            
        inventory_df = pd.DataFrame(inventory_data)
        st.dataframe(inventory_df, use_container_width=True)
        
        # Inventory value
        total_value = inventory_system.get_inventory_value()
        st.metric("Total Inventory Value", f"${total_value}")
        
        # Low stock alerts
        low_stock_items = [item for item, data in inventory_system.known_items.items() 
                         if 0 < data["count"] < data.get("min_stock", 3)]
        
        if low_stock_items and inventory_system.settings["notify_low_stock"]:
            st.warning(f"⚠️ {len(low_stock_items)} items are below minimum stock level!")
            
            for item in low_stock_items:
                item_data = inventory_system.known_items[item]
                st.write(f"- {item.replace('_', ' ').capitalize()}: {item_data['count']} "
                       f"(minimum: {item_data.get('min_stock', 3)})")
                
        # Expiring items
        expiring_items = inventory_system.get_expiring_items()
        if expiring_items:
            st.warning(f"⚠️ {len(expiring_items)} items will expire in the next 3 days!")
            
            for item in expiring_items:
                st.write(f"- {item['item']}: {item['count']} units expire in {item['days_left']} days")
        
        # Inventory visualization
        st.subheader("Inventory Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of current inventory
            fig, ax = plt.subplots(figsize=(10, 6))
            items = [item.replace("_", " ").capitalize() for item in inventory_system.known_items.keys()]
            counts = [data["count"] for data in inventory_system.known_items.values()]
            min_stocks = [data.get("min_stock", 3) for data in inventory_system.known_items.values()]
            
            x = range(len(items))
            bar_width = 0.35
            
            ax.bar(x, counts, bar_width, label='Current Stock')
            ax.bar([i + bar_width for i in x], min_stocks, bar_width, label='Minimum Stock')
            
            ax.set_xlabel('Items')
            ax.set_ylabel('Count')
            ax.set_title('Current Inventory vs. Minimum Stock')
            ax.set_xticks([i + bar_width/2 for i in x])
            ax.set_xticklabels(items, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            # Pie chart of inventory value distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            items = [item.replace("_", " ").capitalize() for item, data in inventory_system.known_items.items() 
                  if data["count"] > 0]
            values = [data["count"] * data.get("unit_price", 0) for item, data in inventory_system.known_items.items() 
                    if data["count"] > 0]
            
            if values:  # Check if there are any values to plot
                ax.pie(values, labels=items, autopct='%1.1f%%', startangle=90)
                ax.set_title('Inventory Value Distribution')
                ax.axis('equal')  # Equal aspect ratio ensures pie is circular
                st.pyplot(fig)
            else:
                st.info("No inventory value to display.")
        
        # Historical inventory trends if available
        if len(inventory_system.inventory_history["timestamps"]) > 1:
            st.subheader("Inventory Trends")
            
            # Convert timestamps to dates for plotting
            dates = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").date() 
                   for ts in inventory_system.inventory_history["timestamps"]]
            
            # Create line chart of inventory history
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for item, counts in inventory_system.inventory_history["data"].items():
                if len(counts) > 0:  # Only plot if we have data
                    ax.plot(dates[-30:], counts[-30:], marker='o', label=item.replace("_", " ").capitalize())
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            ax.set_title('30-Day Inventory History')
            ax.legend()
            ax.grid(True)
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        
    with tab3:
        st.header("Inventory Reports")
        
        # Report options
        report_type = st.selectbox("Select Report Type", 
                                 ["Full Inventory Report", "Low Stock Report", "Expiration Report",
                                 "Purchase Order"])
        
        if report_type == "Full Inventory Report":
            report = inventory_system.generate_inventory_report()
            st.markdown(report)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Report (Markdown)"):
                    # Save report to file
                    report_file = "inventory_report.md"
                    with open(report_file, "w") as f:
                        f.write(report)
                    
                    # Generate download link
                    st.markdown(get_binary_file_downloader_html(report_file, 'Download Report (Markdown)'), 
                                unsafe_allow_html=True)
            
            with col2:
                if st.button("Export Report (CSV)"):
                    # Create DataFrame from inventory
                    export_data = []
                    for item, data in inventory_system.known_items.items():
                        export_data.append({
                            "Item": item.replace("_", " ").capitalize(),
                            "Count": data["count"],
                            "Min_Stock": data.get("min_stock", 3),
                            "Unit_Price": data.get("unit_price", 0),
                            "Total_Value": data["count"] * data.get("unit_price", 0),
                            "Supplier": data.get("supplier", "Unknown"),
                            "Category": data.get("category", "Unknown"),
                            "Last_Updated": data.get("last_updated", "N/A"),
                            "Expiry_Date": data.get("expiry_date", "N/A")
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv_file = "inventory_export.csv"
                    export_df.to_csv(csv_file, index=False)
                    
                    # Generate download link
                    st.markdown(get_binary_file_downloader_html(csv_file, 'Download Report (CSV)'), 
                                unsafe_allow_html=True)
                                
        elif report_type == "Low Stock Report":
            st.subheader("Low Stock Report")
            
            # Get items below minimum stock
            low_stock_items = [(item.replace("_", " ").capitalize(), 
                              data["count"], 
                              data.get("min_stock", 3),
                              data.get("supplier", "Unknown"),
                              (data.get("min_stock", 3) - data["count"]) * data.get("unit_price", 0)) 
                             for item, data in inventory_system.known_items.items() 
                             if data["count"] < data.get("min_stock", 3)]
            
            if low_stock_items:
                low_stock_df = pd.DataFrame(low_stock_items, 
                                          columns=["Item", "Current Stock", "Minimum Stock", 
                                                 "Supplier", "Restock Cost"])
                st.dataframe(low_stock_df, use_container_width=True)
                
                # Calculate totals
                total_items_needed = sum(min_stock - current 
                                       for _, current, min_stock, _, _ in low_stock_items)
                total_restock_cost = sum(cost for _, _, _, _, cost in low_stock_items)
                
                st.metric("Total Items Needed", total_items_needed)
                st.metric("Total Restock Cost", f"${total_restock_cost:.2f}")
            else:
                st.success("No items are below minimum stock levels.")
                
        elif report_type == "Expiration Report":
            st.subheader("Expiration Report")
            
            # Days threshold for expiration warning
            days_threshold = st.slider("Days Threshold", 1, 30, 7)
            
            # Get items expiring within threshold
            expiring_items = inventory_system.get_expiring_items(days_threshold)
            
            if expiring_items:
                expiring_df = pd.DataFrame(expiring_items)
                st.dataframe(expiring_df, use_container_width=True)
                
                # Calculate potential loss
                total_expiring = sum(item["count"] for item in expiring_items)
                total_loss = sum(item["count"] * 
                               inventory_system.known_items[item["item"].lower().replace(" ", "_")].get("unit_price", 0)
                              for item in expiring_items)
                
                st.metric("Total Expiring Items", total_expiring)
                st.metric("Potential Loss", f"${total_loss:.2f}")
                
                # Show expiration chart
                fig, ax = plt.subplots(figsize=(10, 6))
                items = [item["item"] for item in expiring_items]
                days_left = [item["days_left"] for item in expiring_items]
                counts = [item["count"] for item in expiring_items]
                
                # Create scatter plot with size based on count
                scatter = ax.scatter(items, days_left, s=[count * 20 for count in counts], alpha=0.6)
                
                ax.set_xlabel('Items')
                ax.set_ylabel('Days Until Expiry')
                ax.set_title('Expiring Items Overview')
                ax.grid(True)
                
                # Add a horizontal line at immediate expiry (0 days)
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.success(f"No items are expiring within the next {days_threshold} days.")
                
        elif report_type == "Purchase Order":
            st.subheader("Generate Purchase Order")
            
            # Get items to reorder
            reorder_items = inventory_system.get_items_to_reorder()
            
            if reorder_items:
                # Show items that need to be reordered
                reorder_df = pd.DataFrame(reorder_items)
                st.dataframe(reorder_df, use_container_width=True)
                
                # Calculate total cost
                total_cost = sum(item["cost"] for item in reorder_items)
                st.metric("Total Purchase Order Cost", f"${total_cost:.2f}")
                
                # Generate and display PO
                if st.button("Generate Purchase Order"):
                    po_text = inventory_system.auto_generate_purchase_order()
                    st.text_area("Purchase Order", po_text, height=400)
                    
                    # Export PO option
                    po_file = "purchase_order.txt"
                    with open(po_file, "w") as f:
                        f.write(po_text)
                    
                    st.markdown(get_binary_file_downloader_html(po_file, 'Download Purchase Order'), 
                                unsafe_allow_html=True)
            else:
                st.success("No items need to be reordered at this time.")
    
    with tab4:
        st.header("Machine Learning Predictions")
        
        if inventory_system.settings["advanced_features"]:
            # Model training section
            st.subheader("Inventory Prediction Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Train/Update Prediction Model"):
                    with st.spinner("Training model..."):
                        success, message = inventory_system.train_prediction_model()
                        if success:
                            st.success(message)
                        else:
                            st.warning(message)
            
            # Show model status
            if inventory_system.prediction_model is not None:
                st.success("Prediction model is active")
                
                # Days ahead for prediction
                prediction_days = st.slider("Prediction Horizon (Days)", 1, 30, 7)
                
                # Make predictions
                predictions = {}
                for item in inventory_system.known_items:
                    pred = inventory_system.predict_future_inventory(item, prediction_days)
                    if pred is not None:
                        current = inventory_system.known_items[item]["count"]
                        predictions[item] = {
                            "current": current,
                            "predicted": pred,
                            "change": pred - current,
                            "percent_change": ((pred - current) / current * 100) if current > 0 else 0
                        }
                
                if predictions:
                    # Create prediction dataframe
                    pred_data = []
                    for item, data in predictions.items():
                        pred_data.append({
                            "Item": item.replace("_", " ").capitalize(),
                            "Current Count": data["current"],
                            f"Predicted Count ({prediction_days} days)": round(data["predicted"], 1),
                            "Absolute Change": round(data["change"], 1),
                            "Percent Change": f"{round(data['percent_change'], 1)}%"
                        })
                    
                    pred_df = pd.DataFrame(pred_data)
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Prediction visualization
                    st.subheader(f"Predicted Inventory in {prediction_days} Days")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    items = [item.replace("_", " ").capitalize() for item in predictions.keys()]
                    current_vals = [data["current"] for data in predictions.values()]
                    predicted_vals = [data["predicted"] for data in predictions.values()]
                    
                    x = range(len(items))
                    bar_width = 0.35
                    
                    ax.bar(x, current_vals, bar_width, label='Current')
                    ax.bar([i + bar_width for i in x], predicted_vals, bar_width, label='Predicted')
                    
                    # Add arrows showing trend
                    for i, (item, data) in enumerate(predictions.items()):
                        if abs(data["change"]) > 0.5:  # Only show significant changes
                            color = 'green' if data["change"] > 0 else 'red'
                            ax.annotate('', xy=(i + bar_width/2, max(data["current"], data["predicted"]) + 1), 
                                      xytext=(i + bar_width/2, min(data["current"], data["predicted"]) + 1),
                                      arrowprops=dict(arrowstyle='->',
                                                    lw=2,
                                                    color=color))
                    
                    ax.set_xlabel('Items')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Current vs. Predicted Inventory ({prediction_days} days ahead)')
                    ax.set_xticks([i + bar_width/2 for i in x])
                    ax.set_xticklabels(items, rotation=45, ha='right')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Recommendations based on predictions
                    st.subheader("AI Recommendations")
                    
                    # Items predicted to go below minimum stock
                    at_risk_items = []
                    for item, data in predictions.items():
                        min_stock = inventory_system.known_items[item].get("min_stock", 3)
                        if data["predicted"] < min_stock and data["current"] >= min_stock:
                            days_to_min = round((min_stock - data["current"]) / 
                                             (data["change"] / prediction_days)) if data["change"] < 0 else None
                            at_risk_items.append({
                                "item": item.replace("_", " ").capitalize(),
                                "current": data["current"],
                                "predicted": round(data["predicted"], 1),
                                "min_stock": min_stock,
                                "days_to_min": days_to_min
                            })
                    
                    if at_risk_items:
                        st.warning(f"⚠️ {len(at_risk_items)} items are at risk of falling below minimum stock")
                        
                        for item in at_risk_items:
                            if item["days_to_min"]:
                                st.write(f"- {item['item']} will fall below minimum stock in approximately "
                                      f"{item['days_to_min']} days")
                            else:
                                st.write(f"- {item['item']} is predicted to fall below minimum stock")
                        
                        # Early reorder recommendation
                        if st.button("Generate Early Reorder Plan"):
                            st.write("### Early Reorder Plan:")
                            for item in at_risk_items:
                                orig_name = item["item"].lower().replace(" ", "_")
                                supplier = inventory_system.known_items[orig_name].get("supplier", "Unknown")
                                unit_price = inventory_system.known_items[orig_name].get("unit_price", 0)
                                reorder_amount = max(1, round((item["min_stock"] - item["predicted"]) * 1.5))
                                cost = reorder_amount * unit_price
                                
                                st.write(f"* Order {reorder_amount} units of {item['item']} from {supplier} (${cost:.2f})")
                    else:
                        st.success("No items are predicted to fall below minimum stock levels.")
            else:
                st.warning("Prediction model is not available. Please train the model first.")
                
                # Show data requirements
                st.write("### Data requirements for prediction model:")
                st.write("- At least 10 historical inventory records")
                st.write("- Regular inventory updates")
        else:
            st.info("Advanced analytics features are disabled. Enable them in System Features settings.")


if __name__ == "__main__":
    app()