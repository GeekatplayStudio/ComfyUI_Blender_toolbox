import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Geekatplay.KeyManager",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "Geekatplay_ApiKey_Manager") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				
				// Create "Add/Update Key" Button
				this.addWidget("button", "Save New Key", "Save", () => {
					const name = prompt("Enter Service/Key Name (e.g. Tripo3D):");
					if (!name) return;
					
					const value = prompt("Enter API Key Value:");
					if (!value) return;
					
					// Send to backend
					api.fetchApi("/geekatplay/save_key", {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ name, value }),
					}).then(async (resp) => {
						if (resp.ok) {
							alert("Key Saved! Please refresh the page to update the dropdown.");
						} else {
							alert("Error saving key: " + resp.statusText);
						}
					});
				});

                // Create "Delete Key" Button
				this.addWidget("button", "Delete Selected Key", "Delete", () => {
                    const confirmDelete = confirm("Are you sure you want to delete the currently selected key?");
                    if (!confirmDelete) return;

                    // Get current value from the dropdown widget (index 0)
                    const selectedKey = this.widgets[0].value;

					// Send to backend
					api.fetchApi("/geekatplay/delete_key", {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ name: selectedKey }),
					}).then(async (resp) => {
						if (resp.ok) {
							alert("Key Deleted! Please refresh the page.");
						} else {
							alert("Error deleting key: " + resp.statusText);
						}
					});
				});

				return r;
			};
		}
	},
});
