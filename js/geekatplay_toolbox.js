import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Geekatplay.3DToolbox",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "GapPauser") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				const btn = this.addWidget("button", "Continue", null, () => {
					// Call API to unblock
					api.fetchApi("/geekatplay/continue", {
						method: "POST",
						body: JSON.stringify({ node_id: this.id }),
					}).then(async (resp) => {
						if (resp.status === 200) {
							// Visual feedback
							btn.name = "Resumed";
							setTimeout(() => { btn.name = "Continue"; this.setDirtyCanvas(true); }, 1000);
						} else {
							alert("Node is not waiting or error.");
						}
					});
				});
			};
		}

		if (nodeData.name === "GapStringViewer") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				if (message && message.text) {
                    // Update the first widget if it exists and is a text widget, or add it
                    const text = message.text[0];
                    if (this.widgets && this.widgets.length > 0) {
                        this.widgets[0].value = text;
                    } else {
                        // Create widget if missing (though python def should create it via input)
                        // Actually input is forceInput, so likely converted to slot.
                        // We want a display widget.
                        const w = this.addWidget("text", "Text", text, () => {}, { multiline: true });
                        w.inputEl.readOnly = true;
                    }
                    this.onResize?.(this.size);
				}
			};
		}
        
        if (nodeData.name === "GapGroupManager") {
            // Logic to populate toggles
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                this.addWidget("button", "Refresh Groups", null, () => {
                    this.refreshGroupWidgets();
                });
            }

            nodeType.prototype.refreshGroupWidgets = function() {
                // Remove existing toggles (heuristic: type toggle)
                if (this.widgets) {
                     this.widgets = this.widgets.filter(w => w.type === "button");
                }
                
                const groups = app.graph._groups;
                if (!groups) return;

                groups.forEach(group => {
                    this.addWidget("toggle", group.title, true, (value) => {
                        this.toggleGroup(group, value);
                    });
                });
                this.setDirtyCanvas(true);
            };

            nodeType.prototype.toggleGroup = function(group, active) {
                 const nodes = app.graph._nodes;
                 const groupRect = group.getBounding(); // [x, y, w, h]
                 
                 nodes.forEach(node => {
                     // Simple bbox check
                     if (node.pos[0] >= groupRect[0] && 
                         node.pos[1] >= groupRect[1] && 
                         node.pos[0] + node.size[0] <= groupRect[0] + groupRect[2] && 
                         node.pos[1] + node.size[1] <= groupRect[1] + groupRect[3]) {
                             
                             // 0 = Always, 2 = Mute/Bypass (Actually Comfy uses 2 for Mute, 4 for Bypass usually? Or 2 bypass?)
                             // ComfyUI: 0: Always, 1: Never, 2: Bypass, 3: ?
                             // Let's check LiteGraph/Comfy documentation mentally.
                             // setMode(0) -> Always. setMode(2) -> Mute/Never.
                             // Actually "Bypass" is mode 4. "Mute" is mode 2.
                             // PRD says "Bypass or Mute". I'll use Mute (2) as it stops execution.
                             node.mode = active ? 0 : 2; 
                     }
                 });
            }
        }

        if (nodeData.name === "GapVisualComparator") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);
                this.setSize([400, 300]);
                
				this.compData = {
                    imgA: new Image(),
                    imgB: new Image(),
                    loadedA: false,
                    loadedB: false,
                    slider: 0.5,
                    zoom: 1.0,
                    panX: 0,
                    panY: 0,
                    isDraggingSlider: false,
                    isDraggingPan: false,
                    lastMouse: [0, 0]
                };
                
                const redraw = () => { this.setDirtyCanvas(true); };
                this.compData.imgA.onload = () => { this.compData.loadedA = true; redraw(); };
                this.compData.imgB.onload = () => { this.compData.loadedB = true; redraw(); };
                
                 this.addWidget("slider", "Zoom", 1.0, (v) => {
                     this.compData.zoom = v;
                     this.setDirtyCanvas(true);
                 }, { min: 0.1, max: 10.0, step: 0.1 });
			};

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
                if (message && message.comparison_data) {
                    const dA = message.comparison_data[0];
                    const dB = message.comparison_data[1];
                    const getUrl = (d) => `./view?filename=${d.filename}&subfolder=${d.subfolder}&type=${d.type}`;
                    
                    this.compData.loadedA = false;
                    this.compData.loadedB = false;
                    this.compData.imgA.src = getUrl(dA);
                    this.compData.imgB.src = getUrl(dB);
                }
            };
            
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (!this.compData) return;
                
                const margin = 10;
                const top = 50; 
                const w = this.size[0] - margin*2;
                const h = this.size[1] - margin - top;
                
                 if (!this.compData.loadedA || !this.compData.loadedB) {
                    ctx.fillStyle = "#888";
                    ctx.fillText("Waiting for images...", margin, top + 20);
                    return;
                }
                
                ctx.save();
                ctx.beginPath();
                ctx.rect(margin, top, w, h);
                ctx.clip();
                
                const imgW = this.compData.imgA.width;
                const imgH = this.compData.imgA.height;
                const scaleX = w / imgW;
                const scaleY = h / imgH;
                const scale = Math.min(scaleX, scaleY) * this.compData.zoom;
                
                const drawW = imgW * scale;
                const drawH = imgH * scale;
                
                const centerX = margin + w/2 + this.compData.panX;
                const centerY = top + h/2 + this.compData.panY;
                const x = centerX - drawW/2;
                const y = centerY - drawH/2;
                
                const sliderX_rel = w * this.compData.slider;
                const sliderX_abs = margin + sliderX_rel;

                // Left
                ctx.save();
                ctx.beginPath();
                ctx.rect(margin, top, sliderX_rel, h);
                ctx.clip();
                ctx.drawImage(this.compData.imgA, x, y, drawW, drawH);
                ctx.restore();

                // Right
                ctx.save();
                ctx.beginPath();
                ctx.rect(sliderX_abs, top, w - sliderX_rel, h);
                ctx.clip();
                ctx.drawImage(this.compData.imgB, x, y, drawW, drawH);
                ctx.restore();
                
                // Line
                ctx.beginPath();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 2;
                ctx.moveTo(sliderX_abs, top);
                ctx.lineTo(sliderX_abs, top + h);
                ctx.stroke();
                
                ctx.fillStyle = "#fff";
                ctx.beginPath();
                ctx.arc(sliderX_abs, top + h/2, 6, 0, Math.PI*2);
                ctx.fill();
                
                ctx.restore();
            };
            
            nodeType.prototype.onMouseDown = function(e, pos) {
                const x = pos[0];
                const y = pos[1];
                const margin = 10;
                const top = 50;
                const w = this.size[0] - margin*2;
                const h = this.size[1] - margin - top;
                
                 if (y < top || y > top+h || x < margin || x > margin+w) return false;

                const sliderX_abs = margin + w * this.compData.slider;
                if (Math.abs(x - sliderX_abs) < 15) {
                    this.compData.isDraggingSlider = true;
                } else {
                    this.compData.isDraggingPan = true;
                    this.compData.lastMouse = [e.canvasX, e.canvasY];
                }
                return true;
            };
            
            nodeType.prototype.onMouseMove = function(e, pos) {
                if (this.compData.isDraggingSlider) {
                     const margin = 10;
                     const w = this.size[0] - margin*2;
                     let val = (pos[0] - margin) / w;
                     this.compData.slider = Math.max(0, Math.min(1, val));
                     this.setDirtyCanvas(true);
                     return true;
                }
                if (this.compData.isDraggingPan) {
                    const dx = e.canvasX - this.compData.lastMouse[0];
                    const dy = e.canvasY - this.compData.lastMouse[1];
                    this.compData.lastMouse = [e.canvasX, e.canvasY];
                    this.compData.panX += dx;
                    this.compData.panY += dy;
                    this.setDirtyCanvas(true);
                    return true;
                }
            };
            
             nodeType.prototype.onMouseUp = function(e, pos) {
                 this.compData.isDraggingSlider = false;
                 this.compData.isDraggingPan = false;
                 return false;
             };
        }
	},
});
