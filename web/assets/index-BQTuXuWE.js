var __defProp=Object.defineProperty;var __name=(target,value)=>__defProp(target,"name",{value,configurable:!0});import{c9 as ComfyDialog,ca as $el,cb as ComfyApp,b as app,j as LiteGraph,b7 as LGraphCanvas,cc as DraggableList,bl as useToastStore,cd as serialise,aF as useNodeDefStore,ce as deserialiseAndCreate,b0 as api,u as useSettingStore,L as LGraphGroup,cf as KeyComboImpl,O as useKeybindingStore,F as useCommandStore,c as LGraphNode,cg as ComfyWidgets,ch as applyTextReplacements,ci as isElectron,c1 as electronAPI,cj as t,ck as showConfirmationDialog,aQ as nextTick}from"./index-_5-xtSVQ.js";import{mergeIfValid,getWidgetConfig,setWidgetConfig}from"./widgetInputs-CfO8cGW3.js";class ClipspaceDialog extends ComfyDialog{static{__name(this,"ClipspaceDialog")}static items=[];static instance=null;static registerButton(name,contextPredicate,callback){const item=$el("button",{type:"button",textContent:name,contextPredicate,onclick:callback});ClipspaceDialog.items.push(item)}static invalidatePreview(){if(ComfyApp.clipspace&&ComfyApp.clipspace.imgs&&ComfyApp.clipspace.imgs.length>0){const img_preview=document.getElementById("clipspace_preview");img_preview&&(img_preview.src=ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex].src,img_preview.style.maxHeight="100%",img_preview.style.maxWidth="100%")}}static invalidate(){if(ClipspaceDialog.instance){const self2=ClipspaceDialog.instance,children=$el("div.comfy-modal-content",[self2.createImgSettings(),...self2.createButtons()]);self2.element?(self2.element.firstChild&&self2.element.removeChild(self2.element.firstChild),self2.element.appendChild(children)):self2.element=$el("div.comfy-modal",{parent:document.body},[children]),self2.element.children[0].children.length<=1&&self2.element.children[0].appendChild($el("p",{},["Unable to find the features to edit content of a format stored in the current Clipspace."])),ClipspaceDialog.invalidatePreview()}}constructor(){super()}createButtons(){const buttons=[];for(let idx in ClipspaceDialog.items){const item=ClipspaceDialog.items[idx];(!item.contextPredicate||item.contextPredicate())&&buttons.push(ClipspaceDialog.items[idx])}return buttons.push($el("button",{type:"button",textContent:"Close",onclick:__name(()=>{this.close()},"onclick")})),buttons}createImgSettings(){if(ComfyApp.clipspace?.imgs){const combo_items=[],imgs=ComfyApp.clipspace.imgs;for(let i=0;i<imgs.length;i++)combo_items.push($el("option",{value:i},[`${i}`]));const combo1=$el("select",{id:"clipspace_img_selector",onchange:__name(event=>{event.target&&ComfyApp.clipspace&&(ComfyApp.clipspace.selectedIndex=event.target.selectedIndex,ClipspaceDialog.invalidatePreview())},"onchange")},combo_items),row1=$el("tr",{},[$el("td",{},[$el("font",{color:"white"},["Select Image"])]),$el("td",{},[combo1])]),combo2=$el("select",{id:"clipspace_img_paste_mode",onchange:__name(event=>{event.target&&ComfyApp.clipspace&&(ComfyApp.clipspace.img_paste_mode=event.target.value)},"onchange")},[$el("option",{value:"selected"},"selected"),$el("option",{value:"all"},"all")]);combo2.value=ComfyApp.clipspace.img_paste_mode;const row2=$el("tr",{},[$el("td",{},[$el("font",{color:"white"},["Paste Mode"])]),$el("td",{},[combo2])]),td2=$el("td",{align:"center",width:"100px",height:"100px",colSpan:"2"},[$el("img",{id:"clipspace_preview",ondragstart:__name(()=>!1,"ondragstart")},[])]),row3=$el("tr",{},[td2]);return $el("table",{},[row1,row2,row3])}else return[]}createImgPreview(){return ComfyApp.clipspace?.imgs?$el("img",{id:"clipspace_preview",ondragstart:__name(()=>!1,"ondragstart")}):[]}show(){const img_preview=document.getElementById("clipspace_preview");ClipspaceDialog.invalidate(),this.element.style.display="block"}}app.registerExtension({name:"Comfy.Clipspace",init(app2){app2.openClipspace=function(){ClipspaceDialog.instance||(ClipspaceDialog.instance=new ClipspaceDialog,ComfyApp.clipspace_invalidate_handler=ClipspaceDialog.invalidate),ComfyApp.clipspace?ClipspaceDialog.instance.show():app2.ui.dialog.show("Clipspace is Empty!")}}});window.comfyAPI=window.comfyAPI||{};window.comfyAPI.clipspace=window.comfyAPI.clipspace||{};window.comfyAPI.clipspace.ClipspaceDialog=ClipspaceDialog;const ext$1={name:"Comfy.ContextMenuFilter",init(){const ctxMenu=LiteGraph.ContextMenu;LiteGraph.ContextMenu=function(values,options){const ctx=new ctxMenu(values,options);if(options?.className==="dark"&&values?.length>4){const filter=document.createElement("input");filter.classList.add("comfy-context-menu-filter"),filter.placeholder="Filter list",ctx.root.prepend(filter);const items=Array.from(ctx.root.querySelectorAll(".litemenu-entry"));let displayedItems=[...items],itemCount=displayedItems.length;requestAnimationFrame(()=>{const clickedComboValue=LGraphCanvas.active_canvas.current_node?.widgets?.filter(w=>w.type==="combo"&&w.options.values?.length===values.length).find(w=>w.options.values?.every((v,i)=>v===values[i]))?.value;let selectedIndex=clickedComboValue?values.findIndex(v=>v===clickedComboValue):0;selectedIndex<0&&(selectedIndex=0);let selectedItem=displayedItems[selectedIndex];updateSelected();function updateSelected(){selectedItem?.style.setProperty("background-color",""),selectedItem?.style.setProperty("color",""),selectedItem=displayedItems[selectedIndex],selectedItem?.style.setProperty("background-color","#ccc","important"),selectedItem?.style.setProperty("color","#000","important")}__name(updateSelected,"updateSelected");const positionList=__name(()=>{if(ctx.root.getBoundingClientRect().top<0){const scale=1-ctx.root.getBoundingClientRect().height/ctx.root.clientHeight,shift=ctx.root.clientHeight*scale/2;ctx.root.style.top=-shift+"px"}},"positionList");filter.addEventListener("keydown",event=>{switch(event.key){case"ArrowUp":event.preventDefault(),selectedIndex===0?selectedIndex=itemCount-1:selectedIndex--,updateSelected();break;case"ArrowRight":event.preventDefault(),selectedIndex=itemCount-1,updateSelected();break;case"ArrowDown":event.preventDefault(),selectedIndex===itemCount-1?selectedIndex=0:selectedIndex++,updateSelected();break;case"ArrowLeft":event.preventDefault(),selectedIndex=0,updateSelected();break;case"Enter":selectedItem?.click();break;case"Escape":ctx.close();break}}),filter.addEventListener("input",()=>{const term=filter.value.toLocaleLowerCase();if(displayedItems=items.filter(item=>{const isVisible=!term||item.textContent?.toLocaleLowerCase().includes(term);return item.style.display=isVisible?"block":"none",isVisible}),selectedIndex=0,displayedItems.includes(selectedItem)&&(selectedIndex=displayedItems.findIndex(d=>d===selectedItem)),itemCount=displayedItems.length,updateSelected(),options.event){let top=options.event.clientY-10;const bodyRect=document.body.getBoundingClientRect(),rootRect=ctx.root.getBoundingClientRect();bodyRect.height&&top>bodyRect.height-rootRect.height-10&&(top=Math.max(0,bodyRect.height-rootRect.height-10)),ctx.root.style.top=top+"px",positionList()}}),requestAnimationFrame(()=>{filter.focus(),positionList()})})}return ctx},LiteGraph.ContextMenu.prototype=ctxMenu.prototype}};app.registerExtension(ext$1);function stripComments(str){return str.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g,"")}__name(stripComments,"stripComments");app.registerExtension({name:"Comfy.DynamicPrompts",nodeCreated(node){if(node.widgets){const widgets=node.widgets.filter(n=>n.dynamicPrompts);for(const widget of widgets)widget.serializeValue=(workflowNode,widgetIndex)=>{let prompt2=stripComments(widget.value);for(;prompt2.replace("\\{","").includes("{")&&prompt2.replace("\\}","").includes("}");){const startIndex=prompt2.replace("\\{","00").indexOf("{"),endIndex=prompt2.replace("\\}","00").indexOf("}"),options=prompt2.substring(startIndex+1,endIndex).split("|"),randomIndex=Math.floor(Math.random()*options.length),randomOption=options[randomIndex];prompt2=prompt2.substring(0,startIndex)+randomOption+prompt2.substring(endIndex+1)}return workflowNode?.widgets_values&&(workflowNode.widgets_values[widgetIndex]=prompt2),prompt2}}}});app.registerExtension({name:"Comfy.EditAttention",init(){const editAttentionDelta=app.ui.settings.addSetting({id:"Comfy.EditAttention.Delta",name:"Ctrl+up/down precision",type:"slider",attrs:{min:.01,max:.5,step:.01},defaultValue:.05});function incrementWeight(weight,delta){const floatWeight=parseFloat(weight);if(isNaN(floatWeight))return weight;const newWeight=floatWeight+delta;return String(Number(newWeight.toFixed(10)))}__name(incrementWeight,"incrementWeight");function findNearestEnclosure(text,cursorPos){let start=cursorPos,end=cursorPos,openCount=0,closeCount=0;for(;start>=0&&(start--,!(text[start]==="("&&openCount===closeCount));)text[start]==="("&&openCount++,text[start]===")"&&closeCount++;if(start<0)return null;for(openCount=0,closeCount=0;end<text.length&&!(text[end]===")"&&openCount===closeCount);)text[end]==="("&&openCount++,text[end]===")"&&closeCount++,end++;return end===text.length?null:{start:start+1,end}}__name(findNearestEnclosure,"findNearestEnclosure");function addWeightToParentheses(text){const parenRegex=/^\((.*)\)$/,parenMatch=text.match(parenRegex),floatRegex=/:([+-]?(\d*\.)?\d+([eE][+-]?\d+)?)/,floatMatch=text.match(floatRegex);return parenMatch&&!floatMatch?`(${parenMatch[1]}:1.0)`:text}__name(addWeightToParentheses,"addWeightToParentheses");function editAttention(event){const inputField=event.composedPath()[0],delta=parseFloat(editAttentionDelta.value);if(inputField.tagName!=="TEXTAREA"||!(event.key==="ArrowUp"||event.key==="ArrowDown")||!event.ctrlKey&&!event.metaKey)return;event.preventDefault();let start=inputField.selectionStart,end=inputField.selectionEnd,selectedText=inputField.value.substring(start,end);if(!selectedText){const nearestEnclosure=findNearestEnclosure(inputField.value,start);if(nearestEnclosure)start=nearestEnclosure.start,end=nearestEnclosure.end,selectedText=inputField.value.substring(start,end);else{const delimiters=" .,\\/!?%^*;:{}=-_`~()\r\n	";for(;!delimiters.includes(inputField.value[start-1])&&start>0;)start--;for(;!delimiters.includes(inputField.value[end])&&end<inputField.value.length;)end++;if(selectedText=inputField.value.substring(start,end),!selectedText)return}}selectedText[selectedText.length-1]===" "&&(selectedText=selectedText.substring(0,selectedText.length-1),end-=1),inputField.value[start-1]==="("&&inputField.value[end]===")"&&(start-=1,end+=1,selectedText=inputField.value.substring(start,end)),(selectedText[0]!=="("||selectedText[selectedText.length-1]!==")")&&(selectedText=`(${selectedText})`),selectedText=addWeightToParentheses(selectedText);const weightDelta=event.key==="ArrowUp"?delta:-delta,updatedText=selectedText.replace(/\((.*):([+-]?\d+(?:\.\d+)?)\)/,(match,text,weight)=>(weight=incrementWeight(weight,weightDelta),weight==1?text:`(${text}:${weight})`));inputField.setSelectionRange(start,end),document.execCommand("insertText",!1,updatedText),inputField.setSelectionRange(start,start+updatedText.length)}__name(editAttention,"editAttention"),window.addEventListener("keydown",editAttention)}});const ORDER=Symbol(),PREFIX$1="workflow",SEPARATOR$1=">";function merge(target,source){if(typeof target=="object"&&typeof source=="object")for(const key in source){const sv=source[key];if(typeof sv=="object"){let tv=target[key];tv||(tv=target[key]={}),merge(tv,source[key])}else target[key]=sv}return target}__name(merge,"merge");class ManageGroupDialog extends ComfyDialog{static{__name(this,"ManageGroupDialog")}tabs;selectedNodeIndex;selectedTab="Inputs";selectedGroup;modifications={};nodeItems;app;groupNodeType;groupNodeDef;groupData;innerNodesList;widgetsPage;inputsPage;outputsPage;draggable;get selectedNodeInnerIndex(){return+this.nodeItems[this.selectedNodeIndex].dataset.nodeindex}constructor(app2){super(),this.app=app2,this.element=$el("dialog.comfy-group-manage",{parent:document.body})}changeTab(tab){this.tabs[this.selectedTab].tab.classList.remove("active"),this.tabs[this.selectedTab].page.classList.remove("active"),this.tabs[tab].tab.classList.add("active"),this.tabs[tab].page.classList.add("active"),this.selectedTab=tab}changeNode(index,force){!force&&this.selectedNodeIndex===index||(this.selectedNodeIndex!=null&&this.nodeItems[this.selectedNodeIndex].classList.remove("selected"),this.nodeItems[index].classList.add("selected"),this.selectedNodeIndex=index,!this.buildInputsPage()&&this.selectedTab==="Inputs"&&this.changeTab("Widgets"),!this.buildWidgetsPage()&&this.selectedTab==="Widgets"&&this.changeTab("Outputs"),!this.buildOutputsPage()&&this.selectedTab==="Outputs"&&this.changeTab("Inputs"),this.changeTab(this.selectedTab))}getGroupData(){this.groupNodeType=LiteGraph.registered_node_types[`${PREFIX$1}${SEPARATOR$1}`+this.selectedGroup],this.groupNodeDef=this.groupNodeType.nodeData,this.groupData=GroupNodeHandler.getGroupData(this.groupNodeType)}changeGroup(group,reset=!0){this.selectedGroup=group,this.getGroupData();const nodes=this.groupData.nodeData.nodes;if(this.nodeItems=nodes.map((n,i)=>$el("li.draggable-item",{dataset:{nodeindex:n.index+""},onclick:__name(()=>{this.changeNode(i)},"onclick")},[$el("span.drag-handle"),$el("div",{textContent:n.title??n.type},n.title?$el("span",{textContent:n.type}):[])])),this.innerNodesList.replaceChildren(...this.nodeItems),reset)this.selectedNodeIndex=null,this.changeNode(0);else{let index=this.draggable.getAllItems().findIndex(item=>item.classList.contains("selected"));index===-1&&(index=this.selectedNodeIndex),this.changeNode(index,!0)}const ordered=[...nodes];this.draggable?.dispose(),this.draggable=new DraggableList(this.innerNodesList,"li"),this.draggable.addEventListener("dragend",({detail:{oldPosition,newPosition}})=>{if(oldPosition!==newPosition){ordered.splice(newPosition,0,ordered.splice(oldPosition,1)[0]);for(let i=0;i<ordered.length;i++)this.storeModification({nodeIndex:ordered[i].index,section:ORDER,prop:"order",value:i})}})}storeModification(props){const{nodeIndex,section,prop,value}=props,groupMod=this.modifications[this.selectedGroup]??={},nodesMod=groupMod.nodes??={},nodeMod=nodesMod[nodeIndex??this.selectedNodeInnerIndex]??={},typeMod=nodeMod[section]??={};if(typeof value=="object"){const objMod=typeMod[prop]??={};Object.assign(objMod,value)}else typeMod[prop]=value}getEditElement(section,prop,value,placeholder,checked,checkable=!0){value===placeholder&&(value="");const mods=this.modifications[this.selectedGroup]?.nodes?.[this.selectedNodeInnerIndex]?.[section]?.[prop];return mods&&(mods.name!=null&&(value=mods.name),mods.visible!=null&&(checked=mods.visible)),$el("div",[$el("input",{value,placeholder,type:"text",onchange:__name(e=>{this.storeModification({section,prop,value:{name:e.target.value}})},"onchange")}),$el("label",{textContent:"Visible"},[$el("input",{type:"checkbox",checked,disabled:!checkable,onchange:__name(e=>{this.storeModification({section,prop,value:{visible:!!e.target.checked}})},"onchange")})])])}buildWidgetsPage(){const widgets=this.groupData.oldToNewWidgetMap[this.selectedNodeInnerIndex],items=Object.keys(widgets??{}),config=app.graph.extra.groupNodes[this.selectedGroup].config?.[this.selectedNodeInnerIndex]?.input;return this.widgetsPage.replaceChildren(...items.map(oldName=>this.getEditElement("input",oldName,widgets[oldName],oldName,config?.[oldName]?.visible!==!1))),!!items.length}buildInputsPage(){const inputs=this.groupData.nodeInputs[this.selectedNodeInnerIndex],items=Object.keys(inputs??{}),config=app.graph.extra.groupNodes[this.selectedGroup].config?.[this.selectedNodeInnerIndex]?.input;return this.inputsPage.replaceChildren(...items.map(oldName=>{let value=inputs[oldName];if(value)return this.getEditElement("input",oldName,value,oldName,config?.[oldName]?.visible!==!1)}).filter(Boolean)),!!items.length}buildOutputsPage(){const nodes=this.groupData.nodeData.nodes,innerNodeDef=this.groupData.getNodeDef(nodes[this.selectedNodeInnerIndex]),outputs=innerNodeDef?.output??[],groupOutputs=this.groupData.oldToNewOutputMap[this.selectedNodeInnerIndex],config=app.graph.extra.groupNodes[this.selectedGroup].config?.[this.selectedNodeInnerIndex]?.output,checkable=this.groupData.nodeData.nodes[this.selectedNodeInnerIndex].type!=="PrimitiveNode";return this.outputsPage.replaceChildren(...outputs.map((type2,slot)=>{const groupOutputIndex=groupOutputs?.[slot],oldName=innerNodeDef.output_name?.[slot]??type2;let value=config?.[slot]?.name;const visible=config?.[slot]?.visible||groupOutputIndex!=null;return(!value||value===oldName)&&(value=""),this.getEditElement("output",slot,value,oldName,visible,checkable)}).filter(Boolean)),!!outputs.length}show(type){const groupNodes=Object.keys(app.graph.extra?.groupNodes??{}).sort((a,b)=>a.localeCompare(b));this.innerNodesList=$el("ul.comfy-group-manage-list-items"),this.widgetsPage=$el("section.comfy-group-manage-node-page"),this.inputsPage=$el("section.comfy-group-manage-node-page"),this.outputsPage=$el("section.comfy-group-manage-node-page");const pages=$el("div",[this.widgetsPage,this.inputsPage,this.outputsPage]);this.tabs=[["Inputs",this.inputsPage],["Widgets",this.widgetsPage],["Outputs",this.outputsPage]].reduce((p,[name,page])=>(p[name]={tab:$el("a",{onclick:__name(()=>{this.changeTab(name)},"onclick"),textContent:name}),page},p),{});const outer=$el("div.comfy-group-manage-outer",[$el("header",[$el("h2","Group Nodes"),$el("select",{onchange:__name(e=>{this.changeGroup(e.target.value)},"onchange")},groupNodes.map(g=>$el("option",{textContent:g,selected:`${PREFIX$1}${SEPARATOR$1}`+g===type,value:g})))]),$el("main",[$el("section.comfy-group-manage-list",this.innerNodesList),$el("section.comfy-group-manage-node",[$el("header",Object.values(this.tabs).map(t2=>t2.tab)),pages])]),$el("footer",[$el("button.comfy-btn",{onclick:__name(e=>{if(app.graph.nodes.find(n=>n.type===`${PREFIX$1}${SEPARATOR$1}`+this.selectedGroup)){useToastStore().addAlert("This group node is in use in the current workflow, please first remove these.");return}confirm(`Are you sure you want to remove the node: "${this.selectedGroup}"`)&&(delete app.graph.extra.groupNodes[this.selectedGroup],LiteGraph.unregisterNodeType(`${PREFIX$1}${SEPARATOR$1}`+this.selectedGroup)),this.show()},"onclick")},"Delete Group Node"),$el("button.comfy-btn",{onclick:__name(async()=>{let nodesByType,recreateNodes=[];const types={};for(const g in this.modifications){const type2=app.graph.extra.groupNodes[g];let config=type2.config??={},nodeMods=this.modifications[g]?.nodes;if(nodeMods){const keys=Object.keys(nodeMods);if(nodeMods[keys[0]][ORDER]){const orderedNodes=[],orderedMods={},orderedConfig={};for(const n of keys){const order=nodeMods[n][ORDER].order;orderedNodes[order]=type2.nodes[+n],orderedMods[order]=nodeMods[n],orderedNodes[order].index=order}for(const l of type2.links)l[0]!=null&&(l[0]=type2.nodes[l[0]].index),l[2]!=null&&(l[2]=type2.nodes[l[2]].index);if(type2.external)for(const ext2 of type2.external)ext2[0]=type2.nodes[ext2[0]];for(const id2 of keys)config[id2]&&(orderedConfig[type2.nodes[id2].index]=config[id2]),delete config[id2];type2.nodes=orderedNodes,nodeMods=orderedMods,type2.config=config=orderedConfig}merge(config,nodeMods)}types[g]=type2,nodesByType||(nodesByType=app.graph.nodes.reduce((p,n)=>(p[n.type]??=[],p[n.type].push(n),p),{}));const nodes=nodesByType[`${PREFIX$1}${SEPARATOR$1}`+g];nodes&&recreateNodes.push(...nodes)}await GroupNodeConfig.registerFromWorkflow(types,{});for(const node of recreateNodes)node.recreate();this.modifications={},this.app.graph.setDirtyCanvas(!0,!0),this.changeGroup(this.selectedGroup,!1)},"onclick")},"Save"),$el("button.comfy-btn",{onclick:__name(()=>this.element.close(),"onclick")},"Close")])]);this.element.replaceChildren(outer),this.changeGroup(type?groupNodes.find(g=>`${PREFIX$1}${SEPARATOR$1}`+g===type):groupNodes[0]),this.element.showModal(),this.element.addEventListener("close",()=>{this.draggable?.dispose(),this.element.remove()})}}window.comfyAPI=window.comfyAPI||{};window.comfyAPI.groupNodeManage=window.comfyAPI.groupNodeManage||{};window.comfyAPI.groupNodeManage.ManageGroupDialog=ManageGroupDialog;const GROUP=Symbol(),PREFIX="workflow",SEPARATOR=">",Workflow={InUse:{Free:0,Registered:1,InWorkflow:2},isInUseGroupNode(name){const id2=`${PREFIX}${SEPARATOR}${name}`;return app.graph.extra?.groupNodes?.[name]?app.graph.nodes.find(n=>n.type===id2)?Workflow.InUse.InWorkflow:Workflow.InUse.Registered:Workflow.InUse.Free},storeGroupNode(name,data){let extra=app.graph.extra;extra||(app.graph.extra=extra={});let groupNodes=extra.groupNodes;groupNodes||(extra.groupNodes=groupNodes={}),groupNodes[name]=data}};class GroupNodeBuilder{static{__name(this,"GroupNodeBuilder")}nodes;nodeData;constructor(nodes){this.nodes=nodes}build(){const name=this.getName();if(name)return this.sortNodes(),this.nodeData=this.getNodeData(),Workflow.storeGroupNode(name,this.nodeData),{name,nodeData:this.nodeData}}getName(){const name=prompt("Enter group name");if(!name)return;switch(Workflow.isInUseGroupNode(name)){case Workflow.InUse.InWorkflow:useToastStore().addAlert("An in use group node with this name already exists embedded in this workflow, please remove any instances or use a new name.");return;case Workflow.InUse.Registered:if(!confirm("A group node with this name already exists embedded in this workflow, are you sure you want to overwrite it?"))return;break}return name}sortNodes(){const nodesInOrder=app.graph.computeExecutionOrder(!1);this.nodes=this.nodes.map(node=>({index:nodesInOrder.indexOf(node),node})).sort((a,b)=>a.index-b.index||a.node.id-b.node.id).map(({node})=>node)}getNodeData(){const storeLinkTypes=__name(config=>{for(const link of config.links){const type=app.graph.getNodeById(link[4]).outputs[link[1]].type;link.push(type)}},"storeLinkTypes"),storeExternalLinks=__name(config=>{config.external=[];for(let i=0;i<this.nodes.length;i++){const node=this.nodes[i];if(node.outputs?.length)for(let slot=0;slot<node.outputs.length;slot++){let hasExternal=!1;const output=node.outputs[slot];let type=output.type;if(output.links?.length){for(const l of output.links){const link=app.graph.links[l];if(link&&(type==="*"&&(type=link.type),!app.canvas.selected_nodes[link.target_id])){hasExternal=!0;break}}hasExternal&&config.external.push([i,slot,type])}}}},"storeExternalLinks");try{const serialised=serialise(this.nodes,app.canvas.graph),config=JSON.parse(serialised);return storeLinkTypes(config),storeExternalLinks(config),config}finally{}}}class GroupNodeConfig{static{__name(this,"GroupNodeConfig")}name;nodeData;inputCount;oldToNewOutputMap;newToOldOutputMap;oldToNewInputMap;oldToNewWidgetMap;newToOldWidgetMap;primitiveDefs;widgetToPrimitive;primitiveToWidget;nodeInputs;outputVisibility;nodeDef;inputs;linksFrom;linksTo;externalFrom;constructor(name,nodeData){this.name=name,this.nodeData=nodeData,this.getLinks(),this.inputCount=0,this.oldToNewOutputMap={},this.newToOldOutputMap={},this.oldToNewInputMap={},this.oldToNewWidgetMap={},this.newToOldWidgetMap={},this.primitiveDefs={},this.widgetToPrimitive={},this.primitiveToWidget={},this.nodeInputs={},this.outputVisibility=[]}async registerType(source=PREFIX){this.nodeDef={output:[],output_name:[],output_is_list:[],output_is_hidden:[],name:source+SEPARATOR+this.name,display_name:this.name,category:"group nodes"+(SEPARATOR+source),input:{required:{}},description:`Group node combining ${this.nodeData.nodes.map(n=>n.type).join(", ")}`,python_module:"custom_nodes."+this.name,[GROUP]:this},this.inputs=[];const seenInputs={},seenOutputs={};for(let i=0;i<this.nodeData.nodes.length;i++){const node=this.nodeData.nodes[i];node.index=i,this.processNode(node,seenInputs,seenOutputs)}for(const p of this.#convertedToProcess)p();this.#convertedToProcess=null,await app.registerNodeDef(`${PREFIX}${SEPARATOR}`+this.name,this.nodeDef),useNodeDefStore().addNodeDef(this.nodeDef)}getLinks(){this.linksFrom={},this.linksTo={},this.externalFrom={};for(const l of this.nodeData.links){const[sourceNodeId,sourceNodeSlot,targetNodeId,targetNodeSlot]=l;sourceNodeId!=null&&(this.linksFrom[sourceNodeId]||(this.linksFrom[sourceNodeId]={}),this.linksFrom[sourceNodeId][sourceNodeSlot]||(this.linksFrom[sourceNodeId][sourceNodeSlot]=[]),this.linksFrom[sourceNodeId][sourceNodeSlot].push(l),this.linksTo[targetNodeId]||(this.linksTo[targetNodeId]={}),this.linksTo[targetNodeId][targetNodeSlot]=l)}if(this.nodeData.external)for(const ext2 of this.nodeData.external)this.externalFrom[ext2[0]]?this.externalFrom[ext2[0]][ext2[1]]=ext2[2]:this.externalFrom[ext2[0]]={[ext2[1]]:ext2[2]}}processNode(node,seenInputs,seenOutputs){const def=this.getNodeDef(node);if(!def)return;const inputs={...def.input?.required,...def.input?.optional};this.inputs.push(this.processNodeInputs(node,seenInputs,inputs)),def.output?.length&&this.processNodeOutputs(node,seenOutputs,def)}getNodeDef(node){const def=globalDefs[node.type];if(def)return def;const linksFrom=this.linksFrom[node.index];if(node.type==="PrimitiveNode"){if(!linksFrom)return;let type=linksFrom[0][0][5];if(type==="COMBO"){const source=node.outputs[0].widget.name,fromTypeName=this.nodeData.nodes[linksFrom[0][0][2]].type,fromType=globalDefs[fromTypeName];type=(fromType.input.required[source]??fromType.input.optional[source])[0]}return this.primitiveDefs[node.index]={input:{required:{value:[type,{}]}},output:[type],output_name:[],output_is_list:[]}}else if(node.type==="Reroute"){const linksTo=this.linksTo[node.index];if(linksTo&&linksFrom&&!this.externalFrom[node.index]?.[0])return null;let config={},rerouteType="*";if(linksFrom)for(const[,,id2,slot]of linksFrom[0]){const node2=this.nodeData.nodes[id2],input=node2.inputs[slot];if(rerouteType==="*"&&(rerouteType=input.type),input.widget){const targetDef=globalDefs[node2.type],targetWidget=targetDef.input.required[input.widget.name]??targetDef.input.optional[input.widget.name],widget=[targetWidget[0],config];config=mergeIfValid({widget},targetWidget,!1,null,widget)?.customConfig??config}}else if(linksTo){const[id2,slot]=linksTo[0];rerouteType=this.nodeData.nodes[id2].outputs[slot].type}else{for(const l of this.nodeData.links)if(l[2]===node.index){rerouteType=l[5];break}if(rerouteType==="*"){const t2=this.externalFrom[node.index]?.[0];t2&&(rerouteType=t2)}}return config.forceInput=!0,{input:{required:{[rerouteType]:[rerouteType,config]}},output:[rerouteType],output_name:[],output_is_list:[]}}console.warn("Skipping virtual node "+node.type+" when building group node "+this.name)}getInputConfig(node,inputName,seenInputs,config,extra){const customConfig=this.nodeData.config?.[node.index]?.input?.[inputName];let name=customConfig?.name??node.inputs?.find(inp=>inp.name===inputName)?.label??inputName,key=name,prefix="";return(node.type==="PrimitiveNode"&&node.title||name in seenInputs)&&(prefix=`${node.title??node.type} `,key=name=`${prefix}${inputName}`,name in seenInputs&&(name=`${prefix}${seenInputs[name]} ${inputName}`)),seenInputs[key]=(seenInputs[key]??1)+1,(inputName==="seed"||inputName==="noise_seed")&&(extra||(extra={}),extra.control_after_generate=`${prefix}control_after_generate`),config[0]==="IMAGEUPLOAD"&&(extra||(extra={}),extra.widget=this.oldToNewWidgetMap[node.index]?.[config[1]?.widget??"image"]??"image"),extra&&(config=[config[0],{...config[1],...extra}]),{name,config,customConfig}}processWidgetInputs(inputs,node,inputNames,seenInputs){const slots=[],converted=new Map,widgetMap=this.oldToNewWidgetMap[node.index]={};for(const inputName of inputNames)if(app.getWidgetType(inputs[inputName],inputName)){const convertedIndex=node.inputs?.findIndex(inp=>inp.name===inputName&&inp.widget?.name===inputName);if(convertedIndex>-1)converted.set(convertedIndex,inputName),widgetMap[inputName]=null;else{const{name,config}=this.getInputConfig(node,inputName,seenInputs,inputs[inputName]);this.nodeDef.input.required[name]=config,widgetMap[inputName]=name,this.newToOldWidgetMap[name]={node,inputName}}}else slots.push(inputName);return{converted,slots}}checkPrimitiveConnection(link,inputName,inputs){if(this.nodeData.nodes[link[0]].type==="PrimitiveNode"){const[sourceNodeId,_,targetNodeId,__]=link,primitiveDef=this.primitiveDefs[sourceNodeId],targetWidget=inputs[inputName],primitiveConfig=primitiveDef.input.required.value,config=mergeIfValid({widget:primitiveConfig},targetWidget,!1,null,primitiveConfig);primitiveConfig[1]=config?.customConfig??inputs[inputName][1]?{...inputs[inputName][1]}:{};let name=this.oldToNewWidgetMap[sourceNodeId].value;name=name.substr(0,name.length-6),primitiveConfig[1].control_after_generate=!0,primitiveConfig[1].control_prefix=name;let toPrimitive=this.widgetToPrimitive[targetNodeId];toPrimitive||(toPrimitive=this.widgetToPrimitive[targetNodeId]={}),toPrimitive[inputName]&&toPrimitive[inputName].push(sourceNodeId),toPrimitive[inputName]=sourceNodeId;let toWidget=this.primitiveToWidget[sourceNodeId];toWidget||(toWidget=this.primitiveToWidget[sourceNodeId]=[]),toWidget.push({nodeId:targetNodeId,inputName})}}processInputSlots(inputs,node,slots,linksTo,inputMap,seenInputs){this.nodeInputs[node.index]={};for(let i=0;i<slots.length;i++){const inputName=slots[i];if(linksTo[i]){this.checkPrimitiveConnection(linksTo[i],inputName,inputs);continue}const{name,config,customConfig}=this.getInputConfig(node,inputName,seenInputs,inputs[inputName]);this.nodeInputs[node.index][inputName]=name,customConfig?.visible!==!1&&(this.nodeDef.input.required[name]=config,inputMap[i]=this.inputCount++)}}processConvertedWidgets(inputs,node,slots,converted,linksTo,inputMap,seenInputs){const convertedSlots=[...converted.keys()].sort().map(k=>converted.get(k));for(let i=0;i<convertedSlots.length;i++){const inputName=convertedSlots[i];if(linksTo[slots.length+i]){this.checkPrimitiveConnection(linksTo[slots.length+i],inputName,inputs);continue}const{name,config}=this.getInputConfig(node,inputName,seenInputs,inputs[inputName],{defaultInput:!0});this.nodeDef.input.required[name]=config,this.newToOldWidgetMap[name]={node,inputName},this.oldToNewWidgetMap[node.index]||(this.oldToNewWidgetMap[node.index]={}),this.oldToNewWidgetMap[node.index][inputName]=name,inputMap[slots.length+i]=this.inputCount++}}#convertedToProcess=[];processNodeInputs(node,seenInputs,inputs){const inputMapping=[],inputNames=Object.keys(inputs);if(!inputNames.length)return;const{converted,slots}=this.processWidgetInputs(inputs,node,inputNames,seenInputs),linksTo=this.linksTo[node.index]??{},inputMap=this.oldToNewInputMap[node.index]={};return this.processInputSlots(inputs,node,slots,linksTo,inputMap,seenInputs),this.#convertedToProcess.push(()=>this.processConvertedWidgets(inputs,node,slots,converted,linksTo,inputMap,seenInputs)),inputMapping}processNodeOutputs(node,seenOutputs,def){const oldToNew=this.oldToNewOutputMap[node.index]={};for(let outputId=0;outputId<def.output.length;outputId++){const hasLink=this.linksFrom[node.index]?.[outputId]&&!this.externalFrom[node.index]?.[outputId],customConfig=this.nodeData.config?.[node.index]?.output?.[outputId],visible=customConfig?.visible??!hasLink;if(this.outputVisibility.push(visible),!visible)continue;oldToNew[outputId]=this.nodeDef.output.length,this.newToOldOutputMap[this.nodeDef.output.length]={node,slot:outputId},this.nodeDef.output.push(def.output[outputId]),this.nodeDef.output_is_list.push(def.output_is_list[outputId]);let label=customConfig?.name;if(!label){label=def.output_name?.[outputId]??def.output[outputId];const output=node.outputs.find(o=>o.name===label);output?.label&&(label=output.label)}let name=label;if(name in seenOutputs){const prefix=`${node.title??node.type} `;name=`${prefix}${label}`,name in seenOutputs&&(name=`${prefix}${node.index} ${label}`)}seenOutputs[name]=1,this.nodeDef.output_name.push(name)}}static async registerFromWorkflow(groupNodes,missingNodeTypes){for(const g in groupNodes){const groupData=groupNodes[g];let hasMissing=!1;for(const n of groupData.nodes)n.type in LiteGraph.registered_node_types||(missingNodeTypes.push({type:n.type,hint:` (In group node '${PREFIX}${SEPARATOR}${g}')`}),missingNodeTypes.push({type:`${PREFIX}${SEPARATOR}`+g,action:{text:"Remove from workflow",callback:__name(e=>{delete groupNodes[g],e.target.textContent="Removed",e.target.style.pointerEvents="none",e.target.style.opacity=.7},"callback")}}),hasMissing=!0);if(hasMissing)continue;await new GroupNodeConfig(g,groupData).registerType()}}}class GroupNodeHandler{static{__name(this,"GroupNodeHandler")}node;groupData;innerNodes;constructor(node){this.node=node,this.groupData=node.constructor?.nodeData?.[GROUP],this.node.setInnerNodes=innerNodes=>{this.innerNodes=innerNodes;for(let innerNodeIndex=0;innerNodeIndex<this.innerNodes.length;innerNodeIndex++){const innerNode=this.innerNodes[innerNodeIndex];for(const w of innerNode.widgets??[])w.type==="converted-widget"&&(w.serializeValue=w.origSerializeValue);innerNode.index=innerNodeIndex,innerNode.getInputNode=slot=>{const externalSlot=this.groupData.oldToNewInputMap[innerNode.index]?.[slot];if(externalSlot!=null)return this.node.getInputNode(externalSlot);const innerLink=this.groupData.linksTo[innerNode.index]?.[slot];if(!innerLink)return null;const inputNode=innerNodes[innerLink[0]];return inputNode.type==="PrimitiveNode"?null:inputNode},innerNode.getInputLink=slot=>{const externalSlot=this.groupData.oldToNewInputMap[innerNode.index]?.[slot];if(externalSlot!=null){const linkId=this.node.inputs[externalSlot].link;let link2=app.graph.links[linkId];return link2={...link2,target_id:innerNode.id,target_slot:+slot},link2}let link=this.groupData.linksTo[innerNode.index]?.[slot];return link?(link={origin_id:innerNodes[link[0]].id,origin_slot:link[1],target_id:innerNode.id,target_slot:+slot},link):null}}},this.node.updateLink=link=>{link={...link};const output=this.groupData.newToOldOutputMap[link.origin_slot];let innerNode=this.innerNodes[output.node.index],l;for(;innerNode?.type==="Reroute";)l=innerNode.getInputLink(0),innerNode=innerNode.getInputNode(0);return innerNode?l&&GroupNodeHandler.isGroupNode(innerNode)?innerNode.updateLink(l):(link.origin_id=innerNode.id,link.origin_slot=l?.origin_slot??output.slot,link):null},this.node.getInnerNodes=()=>(this.innerNodes||this.node.setInnerNodes(this.groupData.nodeData.nodes.map((n,i)=>{const innerNode=LiteGraph.createNode(n.type);return innerNode.configure(n),innerNode.id=`${this.node.id}:${i}`,innerNode})),this.updateInnerWidgets(),this.innerNodes),this.node.recreate=async()=>{const id2=this.node.id,sz=this.node.size,nodes=this.node.convertToNodes(),groupNode=LiteGraph.createNode(this.node.type);groupNode.id=id2,groupNode.setInnerNodes(nodes),groupNode[GROUP].populateWidgets(),app.graph.add(groupNode),groupNode.size=[Math.max(groupNode.size[0],sz[0]),Math.max(groupNode.size[1],sz[1])];const nodeData=new GroupNodeBuilder(nodes).getNodeData();return groupNode[GROUP].groupData.nodeData.links=nodeData.links,groupNode[GROUP].replaceNodes(nodes),groupNode},this.node.convertToNodes=()=>{const addInnerNodes=__name(()=>{const c={...this.groupData.nodeData};c.nodes=[...c.nodes];const innerNodes=this.node.getInnerNodes();let ids=[];for(let i=0;i<c.nodes.length;i++){let id2=innerNodes?.[i]?.id;id2==null||isNaN(id2)?id2=void 0:ids.push(id2),c.nodes[i]={...c.nodes[i],id:id2}}deserialiseAndCreate(JSON.stringify(c),app.canvas);const[x,y]=this.node.pos;let top,left;const selectedIds=ids.length?ids:Object.keys(app.canvas.selected_nodes),newNodes=[];for(let i=0;i<selectedIds.length;i++){const id2=selectedIds[i],newNode=app.graph.getNodeById(id2),innerNode=innerNodes[i];if(newNodes.push(newNode),(left==null||newNode.pos[0]<left)&&(left=newNode.pos[0]),(top==null||newNode.pos[1]<top)&&(top=newNode.pos[1]),!newNode.widgets)continue;const map=this.groupData.oldToNewWidgetMap[innerNode.index];if(map){const widgets=Object.keys(map);for(const oldName of widgets){const newName=map[oldName];if(!newName)continue;const widgetIndex=this.node.widgets.findIndex(w=>w.name===newName);if(widgetIndex!==-1)if(innerNode.type==="PrimitiveNode")for(let i2=0;i2<newNode.widgets.length;i2++)newNode.widgets[i2].value=this.node.widgets[widgetIndex+i2].value;else{const outerWidget=this.node.widgets[widgetIndex],newWidget=newNode.widgets.find(w=>w.name===oldName);if(!newWidget)continue;newWidget.value=outerWidget.value;for(let w=0;w<outerWidget.linkedWidgets?.length;w++)newWidget.linkedWidgets[w].value=outerWidget.linkedWidgets[w].value}}}}for(const newNode of newNodes)newNode.pos[0]-=left-x,newNode.pos[1]-=top-y;return{newNodes,selectedIds}},"addInnerNodes"),reconnectInputs=__name(selectedIds=>{for(const innerNodeIndex in this.groupData.oldToNewInputMap){const id2=selectedIds[innerNodeIndex],newNode=app.graph.getNodeById(id2),map=this.groupData.oldToNewInputMap[innerNodeIndex];for(const innerInputId in map){const groupSlotId=map[innerInputId];if(groupSlotId==null)continue;const slot=node.inputs[groupSlotId];if(slot.link==null)continue;const link=app.graph.links[slot.link];if(!link)continue;app.graph.getNodeById(link.origin_id).connect(link.origin_slot,newNode,+innerInputId)}}},"reconnectInputs"),reconnectOutputs=__name(selectedIds=>{for(let groupOutputId=0;groupOutputId<node.outputs?.length;groupOutputId++){const output=node.outputs[groupOutputId];if(!output.links)continue;const links=[...output.links];for(const l of links){const slot=this.groupData.newToOldOutputMap[groupOutputId],link=app.graph.links[l],targetNode=app.graph.getNodeById(link.target_id);app.graph.getNodeById(selectedIds[slot.node.index]).connect(slot.slot,targetNode,link.target_slot)}}},"reconnectOutputs");app.canvas.emitBeforeChange();try{const{newNodes,selectedIds}=addInnerNodes();return reconnectInputs(selectedIds),reconnectOutputs(selectedIds),app.graph.remove(this.node),newNodes}finally{app.canvas.emitAfterChange()}};const getExtraMenuOptions=this.node.getExtraMenuOptions;this.node.getExtraMenuOptions=function(_,options){getExtraMenuOptions?.apply(this,arguments);let optionIndex=options.findIndex(o=>o.content==="Outputs");optionIndex===-1?optionIndex=options.length:optionIndex++,options.splice(optionIndex,0,null,{content:"Convert to nodes",callback:__name(()=>this.convertToNodes(),"callback")},{content:"Manage Group Node",callback:__name(()=>manageGroupNodes(this.type),"callback")})};const onDrawTitleBox=this.node.onDrawTitleBox;this.node.onDrawTitleBox=function(ctx,height,size,scale){onDrawTitleBox?.apply(this,arguments);const fill=ctx.fillStyle;ctx.beginPath(),ctx.rect(11,-height+11,2,2),ctx.rect(14,-height+11,2,2),ctx.rect(17,-height+11,2,2),ctx.rect(11,-height+14,2,2),ctx.rect(14,-height+14,2,2),ctx.rect(17,-height+14,2,2),ctx.rect(11,-height+17,2,2),ctx.rect(14,-height+17,2,2),ctx.rect(17,-height+17,2,2),ctx.fillStyle=this.boxcolor||LiteGraph.NODE_DEFAULT_BOXCOLOR,ctx.fill(),ctx.fillStyle=fill};const onDrawForeground=node.onDrawForeground,groupData=this.groupData.nodeData;node.onDrawForeground=function(ctx){const r=onDrawForeground?.apply?.(this,arguments);if(+app.runningNodeId===this.id&&this.runningInternalNodeId!==null){const n=groupData.nodes[this.runningInternalNodeId];if(!n)return;const message=`Running ${n.title||n.type} (${this.runningInternalNodeId}/${groupData.nodes.length})`;ctx.save(),ctx.font="12px sans-serif";const sz=ctx.measureText(message);ctx.fillStyle=node.boxcolor||LiteGraph.NODE_DEFAULT_BOXCOLOR,ctx.beginPath(),ctx.roundRect(0,-LiteGraph.NODE_TITLE_HEIGHT-20,sz.width+12,20,5),ctx.fill(),ctx.fillStyle="#fff",ctx.fillText(message,6,-LiteGraph.NODE_TITLE_HEIGHT-6),ctx.restore()}};const onExecutionStart=this.node.onExecutionStart;this.node.onExecutionStart=function(){return this.resetExecution=!0,onExecutionStart?.apply(this,arguments)};const self2=this,onNodeCreated=this.node.onNodeCreated;this.node.onNodeCreated=function(){if(!this.widgets)return;const config=self2.groupData.nodeData.config;if(config)for(const n in config){const inputs=config[n]?.input;for(const w in inputs){if(inputs[w].visible!==!1)continue;const widgetName=self2.groupData.oldToNewWidgetMap[n][w],widget=this.widgets.find(w2=>w2.name===widgetName);widget&&(widget.type="hidden",widget.computeSize=()=>[0,-4])}}return onNodeCreated?.apply(this,arguments)};function handleEvent(type,getId,getEvent){const handler=__name(({detail})=>{const id2=getId(detail);if(!id2||app.graph.getNodeById(id2))return;const innerNodeIndex=this.innerNodes?.findIndex(n=>n.id==id2);innerNodeIndex>-1&&(this.node.runningInternalNodeId=innerNodeIndex,api.dispatchCustomEvent(type,getEvent(detail,`${this.node.id}`,this.node)))},"handler");return api.addEventListener(type,handler),handler}__name(handleEvent,"handleEvent");const executing=handleEvent.call(this,"executing",d=>d,(d,id2,node2)=>id2),executed=handleEvent.call(this,"executed",d=>d?.display_node||d?.node,(d,id2,node2)=>({...d,node:id2,display_node:id2,merge:!node2.resetExecution})),onRemoved=node.onRemoved;this.node.onRemoved=function(){onRemoved?.apply(this,arguments),api.removeEventListener("executing",executing),api.removeEventListener("executed",executed)},this.node.refreshComboInNode=defs=>{for(const widgetName in this.groupData.newToOldWidgetMap){const widget=this.node.widgets.find(w=>w.name===widgetName);if(widget?.type==="combo"){const old=this.groupData.newToOldWidgetMap[widgetName],def=defs[old.node.type],input=def?.input?.required?.[old.inputName]??def?.input?.optional?.[old.inputName];if(!input)continue;widget.options.values=input[0],old.inputName!=="image"&&!widget.options.values.includes(widget.value)&&(widget.value=widget.options.values[0],widget.callback(widget.value))}}}}updateInnerWidgets(){for(const newWidgetName in this.groupData.newToOldWidgetMap){const newWidget=this.node.widgets.find(w=>w.name===newWidgetName);if(!newWidget)continue;const newValue=newWidget.value,old=this.groupData.newToOldWidgetMap[newWidgetName];let innerNode=this.innerNodes[old.node.index];if(innerNode.type==="PrimitiveNode"){innerNode.primitiveValue=newValue;const primitiveLinked=this.groupData.primitiveToWidget[old.node.index];for(const linked of primitiveLinked??[]){const widget2=this.innerNodes[linked.nodeId].widgets.find(w=>w.name===linked.inputName);widget2&&(widget2.value=newValue)}continue}else if(innerNode.type==="Reroute"){const rerouteLinks=this.groupData.linksFrom[old.node.index];if(rerouteLinks)for(const[_,,targetNodeId,targetSlot]of rerouteLinks[0]){const node=this.innerNodes[targetNodeId],input=node.inputs[targetSlot];if(input.widget){const widget2=node.widgets?.find(w=>w.name===input.widget.name);widget2&&(widget2.value=newValue)}}}const widget=innerNode.widgets?.find(w=>w.name===old.inputName);widget&&(widget.value=newValue)}}populatePrimitive(node,nodeId,oldName,i,linkedShift){const primitiveId=this.groupData.widgetToPrimitive[nodeId]?.[oldName];if(primitiveId==null)return;const targetWidgetName=this.groupData.oldToNewWidgetMap[primitiveId].value,targetWidgetIndex=this.node.widgets.findIndex(w=>w.name===targetWidgetName);if(targetWidgetIndex>-1){const primitiveNode=this.innerNodes[primitiveId];let len=primitiveNode.widgets.length;len-1!==this.node.widgets[targetWidgetIndex].linkedWidgets?.length&&(len=1);for(let i2=0;i2<len;i2++)this.node.widgets[targetWidgetIndex+i2].value=primitiveNode.widgets[i2].value}return!0}populateReroute(node,nodeId,map){if(node.type!=="Reroute")return;const link=this.groupData.linksFrom[nodeId]?.[0]?.[0];if(!link)return;const[,,targetNodeId,targetNodeSlot]=link,targetNode=this.groupData.nodeData.nodes[targetNodeId],inputs=targetNode.inputs;if(!inputs?.[targetNodeSlot]?.widget)return;const offset=inputs.length-(targetNode.widgets_values?.length??0),v=targetNode.widgets_values?.[targetNodeSlot-offset];if(v==null)return;const widgetName=Object.values(map)[0],widget=this.node.widgets.find(w=>w.name===widgetName);widget&&(widget.value=v)}populateWidgets(){if(this.node.widgets)for(let nodeId=0;nodeId<this.groupData.nodeData.nodes.length;nodeId++){const node=this.groupData.nodeData.nodes[nodeId],map=this.groupData.oldToNewWidgetMap[nodeId]??{},widgets=Object.keys(map);if(!node.widgets_values?.length){this.populateReroute(node,nodeId,map);continue}let linkedShift=0;for(let i=0;i<widgets.length;i++){const oldName=widgets[i],newName=map[oldName],widgetIndex=this.node.widgets.findIndex(w=>w.name===newName),mainWidget=this.node.widgets[widgetIndex];if(this.populatePrimitive(node,nodeId,oldName,i,linkedShift)||widgetIndex===-1){const innerWidget=this.innerNodes[nodeId].widgets?.find(w=>w.name===oldName);linkedShift+=innerWidget?.linkedWidgets?.length??0}if(widgetIndex!==-1){mainWidget.value=node.widgets_values[i+linkedShift];for(let w=0;w<mainWidget.linkedWidgets?.length;w++)this.node.widgets[widgetIndex+w+1].value=node.widgets_values[i+ ++linkedShift]}}}}replaceNodes(nodes){let top,left;for(let i=0;i<nodes.length;i++){const node=nodes[i];(left==null||node.pos[0]<left)&&(left=node.pos[0]),(top==null||node.pos[1]<top)&&(top=node.pos[1]),this.linkOutputs(node,i),app.graph.remove(node)}this.linkInputs(),this.node.pos=[left,top]}linkOutputs(originalNode,nodeId){if(originalNode.outputs)for(const output of originalNode.outputs){if(!output.links)continue;const links=[...output.links];for(const l of links){const link=app.graph.links[l];if(!link)continue;const targetNode=app.graph.getNodeById(link.target_id),newSlot=this.groupData.oldToNewOutputMap[nodeId]?.[link.origin_slot];newSlot!=null&&this.node.connect(newSlot,targetNode,link.target_slot)}}}linkInputs(){for(const link of this.groupData.nodeData.links??[]){const[,originSlot,targetId,targetSlot,actualOriginId]=link,originNode=app.graph.getNodeById(actualOriginId);originNode&&originNode.connect(originSlot,this.node.id,this.groupData.oldToNewInputMap[targetId][targetSlot])}}static getGroupData(node){return(node.nodeData??node.constructor?.nodeData)?.[GROUP]}static isGroupNode(node){return!!node.constructor?.nodeData?.[GROUP]}static async fromNodes(nodes){const builder=new GroupNodeBuilder(nodes),res=builder.build();if(!res)return;const{name,nodeData}=res;await new GroupNodeConfig(name,nodeData).registerType();const groupNode=LiteGraph.createNode(`${PREFIX}${SEPARATOR}${name}`);return groupNode.setInnerNodes(builder.nodes),groupNode[GROUP].populateWidgets(),app.graph.add(groupNode),groupNode[GROUP].replaceNodes(builder.nodes),groupNode}}function addConvertToGroupOptions(){function addConvertOption(options,index){const selected=Object.values(app.canvas.selected_nodes??{}),disabled=selected.length<2||selected.find(n=>GroupNodeHandler.isGroupNode(n));options.splice(index+1,null,{content:"Convert to Group Node",disabled,callback:convertSelectedNodesToGroupNode})}__name(addConvertOption,"addConvertOption");function addManageOption(options,index){const groups=app.graph.extra?.groupNodes,disabled=!groups||!Object.keys(groups).length;options.splice(index+1,null,{content:"Manage Group Nodes",disabled,callback:manageGroupNodes})}__name(addManageOption,"addManageOption");const getCanvasMenuOptions=LGraphCanvas.prototype.getCanvasMenuOptions;LGraphCanvas.prototype.getCanvasMenuOptions=function(){const options=getCanvasMenuOptions.apply(this,arguments),index=options.findIndex(o=>o?.content==="Add Group")+1||options.length;return addConvertOption(options,index),addManageOption(options,index+1),options};const getNodeMenuOptions=LGraphCanvas.prototype.getNodeMenuOptions;LGraphCanvas.prototype.getNodeMenuOptions=function(node){const options=getNodeMenuOptions.apply(this,arguments);if(!GroupNodeHandler.isGroupNode(node)){const index=options.findIndex(o=>o?.content==="Outputs")+1||options.length-1;addConvertOption(options,index)}return options}}__name(addConvertToGroupOptions,"addConvertToGroupOptions");const replaceLegacySeparators=__name(nodes=>{for(const node of nodes)typeof node.type=="string"&&node.type.startsWith("workflow/")&&(node.type=node.type.replace(/^workflow\//,`${PREFIX}${SEPARATOR}`))},"replaceLegacySeparators");async function convertSelectedNodesToGroupNode(){const nodes=Object.values(app.canvas.selected_nodes??{});if(nodes.length===0)throw new Error("No nodes selected");if(nodes.length===1)throw new Error("Please select multiple nodes to convert to group node");if(nodes.some(n=>GroupNodeHandler.isGroupNode(n)))throw new Error("Selected nodes contain a group node");return await GroupNodeHandler.fromNodes(nodes)}__name(convertSelectedNodesToGroupNode,"convertSelectedNodesToGroupNode");function ungroupSelectedGroupNodes(){const nodes=Object.values(app.canvas.selected_nodes??{});for(const node of nodes)GroupNodeHandler.isGroupNode(node)&&node.convertToNodes?.()}__name(ungroupSelectedGroupNodes,"ungroupSelectedGroupNodes");function manageGroupNodes(type){new ManageGroupDialog(app).show(type)}__name(manageGroupNodes,"manageGroupNodes");const id$2="Comfy.GroupNode";let globalDefs;const ext={name:id$2,commands:[{id:"Comfy.GroupNode.ConvertSelectedNodesToGroupNode",label:"Convert selected nodes to group node",icon:"pi pi-sitemap",versionAdded:"1.3.17",function:convertSelectedNodesToGroupNode},{id:"Comfy.GroupNode.UngroupSelectedGroupNodes",label:"Ungroup selected group nodes",icon:"pi pi-sitemap",versionAdded:"1.3.17",function:ungroupSelectedGroupNodes},{id:"Comfy.GroupNode.ManageGroupNodes",label:"Manage group nodes",icon:"pi pi-cog",versionAdded:"1.3.17",function:manageGroupNodes}],keybindings:[{commandId:"Comfy.GroupNode.ConvertSelectedNodesToGroupNode",combo:{alt:!0,key:"g"}},{commandId:"Comfy.GroupNode.UngroupSelectedGroupNodes",combo:{alt:!0,shift:!0,key:"G"}}],setup(){addConvertToGroupOptions()},async beforeConfigureGraph(graphData,missingNodeTypes){const nodes=graphData?.extra?.groupNodes;nodes&&(replaceLegacySeparators(graphData.nodes),await GroupNodeConfig.registerFromWorkflow(nodes,missingNodeTypes))},addCustomNodeDefs(defs){globalDefs=defs},nodeCreated(node){GroupNodeHandler.isGroupNode(node)&&(node[GROUP]=new GroupNodeHandler(node),node.title&&node[GROUP]?.groupData?.nodeData&&Workflow.storeGroupNode(node.title,node[GROUP].groupData.nodeData))},async refreshComboInNodes(defs){Object.assign(globalDefs,defs);const nodes=app.graph.extra?.groupNodes;nodes&&await GroupNodeConfig.registerFromWorkflow(nodes,{})}};app.registerExtension(ext);window.comfyAPI=window.comfyAPI||{};window.comfyAPI.groupNode=window.comfyAPI.groupNode||{};window.comfyAPI.groupNode.GroupNodeConfig=GroupNodeConfig;window.comfyAPI.groupNode.GroupNodeHandler=GroupNodeHandler;function setNodeMode(node,mode){node.mode=mode,node.graph?.change()}__name(setNodeMode,"setNodeMode");function addNodesToGroup(group,items){const padding=useSettingStore().get("Comfy.GroupSelectedNodes.Padding");group.resizeTo([...group.children,...items],padding)}__name(addNodesToGroup,"addNodesToGroup");app.registerExtension({name:"Comfy.GroupOptions",setup(){const orig=LGraphCanvas.prototype.getCanvasMenuOptions;LGraphCanvas.prototype.getCanvasMenuOptions=function(){const options=orig.apply(this,arguments),group=this.graph.getGroupOnPos(this.graph_mouse[0],this.graph_mouse[1]);if(!group)return options.push({content:"Add Group For Selected Nodes",disabled:!this.selectedItems?.size,callback:__name(()=>{const group2=new LGraphGroup;addNodesToGroup(group2,this.selectedItems),this.graph.add(group2),this.graph.change()},"callback")}),options;group.recomputeInsideNodes();const nodesInGroup=group.nodes;if(options.push({content:"Add Selected Nodes To Group",disabled:!this.selectedItems?.size,callback:__name(()=>{addNodesToGroup(group,this.selectedItems),this.graph.change()},"callback")}),nodesInGroup.length===0)return options;options.push(null);let allNodesAreSameMode=!0;for(let i=1;i<nodesInGroup.length;i++)if(nodesInGroup[i].mode!==nodesInGroup[0].mode){allNodesAreSameMode=!1;break}if(options.push({content:"Fit Group To Nodes",callback:__name(()=>{group.recomputeInsideNodes();const padding=useSettingStore().get("Comfy.GroupSelectedNodes.Padding");group.resizeTo(group.children,padding),this.graph.change()},"callback")}),options.push({content:"Select Nodes",callback:__name(()=>{this.selectNodes(nodesInGroup),this.graph.change(),this.canvas.focus()},"callback")}),allNodesAreSameMode)switch(nodesInGroup[0].mode){case 0:options.push({content:"Set Group Nodes to Never",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,2)},"callback")}),options.push({content:"Bypass Group Nodes",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,4)},"callback")});break;case 2:options.push({content:"Set Group Nodes to Always",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,0)},"callback")}),options.push({content:"Bypass Group Nodes",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,4)},"callback")});break;case 4:options.push({content:"Set Group Nodes to Always",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,0)},"callback")}),options.push({content:"Set Group Nodes to Never",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,2)},"callback")});break;default:options.push({content:"Set Group Nodes to Always",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,0)},"callback")}),options.push({content:"Set Group Nodes to Never",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,2)},"callback")}),options.push({content:"Bypass Group Nodes",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,4)},"callback")});break}else options.push({content:"Set Group Nodes to Always",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,0)},"callback")}),options.push({content:"Set Group Nodes to Never",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,2)},"callback")}),options.push({content:"Bypass Group Nodes",callback:__name(()=>{for(const node of nodesInGroup)setNodeMode(node,4)},"callback")});return options}}});const id$1="Comfy.InvertMenuScrolling";app.registerExtension({name:id$1,init(){const ctxMenu=LiteGraph.ContextMenu,replace=__name(()=>{LiteGraph.ContextMenu=function(values,options){return options=options||{},options.scroll_speed?options.scroll_speed*=-1:options.scroll_speed=-.1,ctxMenu.call(this,values,options)},LiteGraph.ContextMenu.prototype=ctxMenu.prototype},"replace");app.ui.settings.addSetting({id:id$1,category:["LiteGraph","Menu","InvertMenuScrolling"],name:"Invert Context Menu Scrolling",type:"boolean",defaultValue:!1,onChange(value){value?replace():LiteGraph.ContextMenu=ctxMenu}})}});app.registerExtension({name:"Comfy.Keybinds",init(){const keybindListener=__name(async function(event){if(!app.vueAppReady)return;const keyCombo=KeyComboImpl.fromEvent(event);if(keyCombo.isModifier)return;const target=event.composedPath()[0];if(!keyCombo.hasModifier&&(target.tagName==="TEXTAREA"||target.tagName==="INPUT"||target.tagName==="SPAN"&&target.classList.contains("property_value")))return;const keybindingStore=useKeybindingStore(),commandStore=useCommandStore(),keybinding=keybindingStore.getKeybinding(keyCombo);if(keybinding&&keybinding.targetSelector!=="#graph-canvas"){event.preventDefault(),await commandStore.execute(keybinding.commandId);return}if(!(event.ctrlKey||event.altKey||event.metaKey)&&event.key==="Escape"){const modals=document.querySelectorAll(".comfy-modal");for(const modal of modals)if(window.getComputedStyle(modal).getPropertyValue("display")!=="none"){modal.style.display="none";break}for(const d of document.querySelectorAll("dialog"))d.close()}},"keybindListener");window.addEventListener("keydown",keybindListener)}});function dataURLToBlob(dataURL){const parts=dataURL.split(";base64,"),contentType=parts[0].split(":")[1],byteString=atob(parts[1]),arrayBuffer=new ArrayBuffer(byteString.length),uint8Array=new Uint8Array(arrayBuffer);for(let i=0;i<byteString.length;i++)uint8Array[i]=byteString.charCodeAt(i);return new Blob([arrayBuffer],{type:contentType})}__name(dataURLToBlob,"dataURLToBlob");function loadImage(imagePath){return new Promise((resolve,reject)=>{const image=new Image;image.onload=function(){resolve(image)},image.src=imagePath})}__name(loadImage,"loadImage");async function uploadMask(filepath,formData){await api.fetchApi("/upload/mask",{method:"POST",body:formData}).then(response=>{}).catch(error=>{console.error("Error:",error)}),ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex]=new Image,ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex].src=api.apiURL("/view?"+new URLSearchParams(filepath).toString()+app.getPreviewFormatParam()+app.getRandParam()),ComfyApp.clipspace.images&&(ComfyApp.clipspace.images[ComfyApp.clipspace.selectedIndex]=filepath),ClipspaceDialog.invalidatePreview()}__name(uploadMask,"uploadMask");function prepare_mask(image,maskCanvas,maskCtx,maskColor){maskCtx.drawImage(image,0,0,maskCanvas.width,maskCanvas.height);const maskData=maskCtx.getImageData(0,0,maskCanvas.width,maskCanvas.height);for(let i=0;i<maskData.data.length;i+=4)maskData.data[i+3]==255?maskData.data[i+3]=0:maskData.data[i+3]=255,maskData.data[i]=maskColor.r,maskData.data[i+1]=maskColor.g,maskData.data[i+2]=maskColor.b;maskCtx.globalCompositeOperation="source-over",maskCtx.putImageData(maskData,0,0)}__name(prepare_mask,"prepare_mask");class MaskEditorDialogOld extends ComfyDialog{static{__name(this,"MaskEditorDialogOld")}static instance=null;static mousedown_x=null;static mousedown_y=null;brush;maskCtx;maskCanvas;brush_size_slider;brush_opacity_slider;colorButton;saveButton;zoom_ratio;pan_x;pan_y;imgCanvas;last_display_style;is_visible;image;handler_registered;brush_slider_input;cursorX;cursorY;mousedown_pan_x;mousedown_pan_y;last_pressure;pointer_type;brush_pointer_type_select;static getInstance(){return MaskEditorDialogOld.instance||(MaskEditorDialogOld.instance=new MaskEditorDialogOld),MaskEditorDialogOld.instance}is_layout_created=!1;constructor(){super(),this.element=$el("div.comfy-modal",{parent:document.body},[$el("div.comfy-modal-content",[...this.createButtons()])])}createButtons(){return[]}createButton(name,callback){var button=document.createElement("button");return button.style.pointerEvents="auto",button.innerText=name,button.addEventListener("click",callback),button}createLeftButton(name,callback){var button=this.createButton(name,callback);return button.style.cssFloat="left",button.style.marginRight="4px",button}createRightButton(name,callback){var button=this.createButton(name,callback);return button.style.cssFloat="right",button.style.marginLeft="4px",button}createLeftSlider(self2,name,callback){const divElement=document.createElement("div");divElement.id="maskeditor-slider",divElement.style.cssFloat="left",divElement.style.fontFamily="sans-serif",divElement.style.marginRight="4px",divElement.style.color="var(--input-text)",divElement.style.backgroundColor="var(--comfy-input-bg)",divElement.style.borderRadius="8px",divElement.style.borderColor="var(--border-color)",divElement.style.borderStyle="solid",divElement.style.fontSize="15px",divElement.style.height="25px",divElement.style.padding="1px 6px",divElement.style.display="flex",divElement.style.position="relative",divElement.style.top="2px",divElement.style.pointerEvents="auto",self2.brush_slider_input=document.createElement("input"),self2.brush_slider_input.setAttribute("type","range"),self2.brush_slider_input.setAttribute("min","1"),self2.brush_slider_input.setAttribute("max","100"),self2.brush_slider_input.setAttribute("value","10");const labelElement=document.createElement("label");return labelElement.textContent=name,divElement.appendChild(labelElement),divElement.appendChild(self2.brush_slider_input),self2.brush_slider_input.addEventListener("change",callback),divElement}createOpacitySlider(self2,name,callback){const divElement=document.createElement("div");divElement.id="maskeditor-opacity-slider",divElement.style.cssFloat="left",divElement.style.fontFamily="sans-serif",divElement.style.marginRight="4px",divElement.style.color="var(--input-text)",divElement.style.backgroundColor="var(--comfy-input-bg)",divElement.style.borderRadius="8px",divElement.style.borderColor="var(--border-color)",divElement.style.borderStyle="solid",divElement.style.fontSize="15px",divElement.style.height="25px",divElement.style.padding="1px 6px",divElement.style.display="flex",divElement.style.position="relative",divElement.style.top="2px",divElement.style.pointerEvents="auto",self2.opacity_slider_input=document.createElement("input"),self2.opacity_slider_input.setAttribute("type","range"),self2.opacity_slider_input.setAttribute("min","0.1"),self2.opacity_slider_input.setAttribute("max","1.0"),self2.opacity_slider_input.setAttribute("step","0.01"),self2.opacity_slider_input.setAttribute("value","0.7");const labelElement=document.createElement("label");return labelElement.textContent=name,divElement.appendChild(labelElement),divElement.appendChild(self2.opacity_slider_input),self2.opacity_slider_input.addEventListener("input",callback),divElement}createPointerTypeSelect(self2){const divElement=document.createElement("div");divElement.id="maskeditor-pointer-type",divElement.style.cssFloat="left",divElement.style.fontFamily="sans-serif",divElement.style.marginRight="4px",divElement.style.color="var(--input-text)",divElement.style.backgroundColor="var(--comfy-input-bg)",divElement.style.borderRadius="8px",divElement.style.borderColor="var(--border-color)",divElement.style.borderStyle="solid",divElement.style.fontSize="15px",divElement.style.height="25px",divElement.style.padding="1px 6px",divElement.style.display="flex",divElement.style.position="relative",divElement.style.top="2px",divElement.style.pointerEvents="auto";const labelElement=document.createElement("label");labelElement.textContent="Pointer Type:";const selectElement=document.createElement("select");selectElement.style.borderRadius="0",selectElement.style.borderColor="transparent",selectElement.style.borderStyle="unset",selectElement.style.fontSize="0.9em";const optionArc=document.createElement("option");optionArc.value="arc",optionArc.text="Circle",optionArc.selected=!0;const optionRect=document.createElement("option");return optionRect.value="rect",optionRect.text="Square",selectElement.appendChild(optionArc),selectElement.appendChild(optionRect),selectElement.addEventListener("change",event=>{const target=event.target;self2.pointer_type=target.value,this.setBrushBorderRadius(self2)}),divElement.appendChild(labelElement),divElement.appendChild(selectElement),divElement}setBrushBorderRadius(self2){self2.pointer_type==="rect"?(this.brush.style.borderRadius="0%",this.brush.style.MozBorderRadius="0%",this.brush.style.WebkitBorderRadius="0%"):(this.brush.style.borderRadius="50%",this.brush.style.MozBorderRadius="50%",this.brush.style.WebkitBorderRadius="50%")}setlayout(imgCanvas,maskCanvas){const self2=this;self2.pointer_type="arc";var bottom_panel=document.createElement("div");bottom_panel.style.position="absolute",bottom_panel.style.bottom="0px",bottom_panel.style.left="20px",bottom_panel.style.right="20px",bottom_panel.style.height="50px",bottom_panel.style.pointerEvents="none";var brush=document.createElement("div");brush.id="brush",brush.style.backgroundColor="transparent",brush.style.outline="1px dashed black",brush.style.boxShadow="0 0 0 1px white",brush.style.position="absolute",brush.style.zIndex="8889",brush.style.pointerEvents="none",this.brush=brush,this.setBrushBorderRadius(self2),this.element.appendChild(imgCanvas),this.element.appendChild(maskCanvas),this.element.appendChild(bottom_panel),document.body.appendChild(brush);var clearButton=this.createLeftButton("Clear",()=>{self2.maskCtx.clearRect(0,0,self2.maskCanvas.width,self2.maskCanvas.height)});this.brush_size_slider=this.createLeftSlider(self2,"Thickness",event=>{self2.brush_size=event.target.value,self2.updateBrushPreview(self2)}),this.brush_opacity_slider=this.createOpacitySlider(self2,"Opacity",event=>{self2.brush_opacity=event.target.value,self2.brush_color_mode!=="negative"&&(self2.maskCanvas.style.opacity=self2.brush_opacity.toString())}),this.brush_pointer_type_select=this.createPointerTypeSelect(self2),this.colorButton=this.createLeftButton(this.getColorButtonText(),()=>{self2.brush_color_mode==="black"?self2.brush_color_mode="white":self2.brush_color_mode==="white"?self2.brush_color_mode="negative":self2.brush_color_mode="black",self2.updateWhenBrushColorModeChanged()});var cancelButton=this.createRightButton("Cancel",()=>{document.removeEventListener("keydown",MaskEditorDialogOld.handleKeyDown),self2.close()});this.saveButton=this.createRightButton("Save",()=>{document.removeEventListener("keydown",MaskEditorDialogOld.handleKeyDown),self2.save()}),this.element.appendChild(imgCanvas),this.element.appendChild(maskCanvas),this.element.appendChild(bottom_panel),bottom_panel.appendChild(clearButton),bottom_panel.appendChild(this.saveButton),bottom_panel.appendChild(cancelButton),bottom_panel.appendChild(this.brush_size_slider),bottom_panel.appendChild(this.brush_opacity_slider),bottom_panel.appendChild(this.brush_pointer_type_select),bottom_panel.appendChild(this.colorButton),imgCanvas.style.position="absolute",maskCanvas.style.position="absolute",imgCanvas.style.top="200",imgCanvas.style.left="0",maskCanvas.style.top=imgCanvas.style.top,maskCanvas.style.left=imgCanvas.style.left;const maskCanvasStyle=this.getMaskCanvasStyle();maskCanvas.style.mixBlendMode=maskCanvasStyle.mixBlendMode,maskCanvas.style.opacity=maskCanvasStyle.opacity.toString()}async show(){if(this.zoom_ratio=1,this.pan_x=0,this.pan_y=0,!this.is_layout_created){const imgCanvas=document.createElement("canvas"),maskCanvas=document.createElement("canvas");imgCanvas.id="imageCanvas",maskCanvas.id="maskCanvas",this.setlayout(imgCanvas,maskCanvas),this.imgCanvas=imgCanvas,this.maskCanvas=maskCanvas,this.maskCtx=maskCanvas.getContext("2d",{willReadFrequently:!0}),this.setEventHandler(maskCanvas),this.is_layout_created=!0;const self2=this,observer=new MutationObserver(function(mutations){mutations.forEach(function(mutation){mutation.type==="attributes"&&mutation.attributeName==="style"&&(self2.last_display_style&&self2.last_display_style!="none"&&self2.element.style.display=="none"&&(self2.brush.style.display="none",ComfyApp.onClipspaceEditorClosed()),self2.last_display_style=self2.element.style.display)})}),config={attributes:!0};observer.observe(this.element,config)}document.addEventListener("keydown",MaskEditorDialogOld.handleKeyDown),ComfyApp.clipspace_return_node?this.saveButton.innerText="Save to node":this.saveButton.innerText="Save",this.saveButton.disabled=!1,this.element.style.display="block",this.element.style.width="85%",this.element.style.margin="0 7.5%",this.element.style.height="100vh",this.element.style.top="50%",this.element.style.left="42%",this.element.style.zIndex="8888",await this.setImages(this.imgCanvas),this.is_visible=!0}isOpened(){return this.element.style.display=="block"}invalidateCanvas(orig_image,mask_image){this.imgCanvas.width=orig_image.width,this.imgCanvas.height=orig_image.height,this.maskCanvas.width=orig_image.width,this.maskCanvas.height=orig_image.height;let imgCtx=this.imgCanvas.getContext("2d",{willReadFrequently:!0}),maskCtx=this.maskCanvas.getContext("2d",{willReadFrequently:!0});imgCtx.drawImage(orig_image,0,0,orig_image.width,orig_image.height),prepare_mask(mask_image,this.maskCanvas,maskCtx,this.getMaskColor())}async setImages(imgCanvas){let self2=this;const imgCtx=imgCanvas.getContext("2d",{willReadFrequently:!0}),maskCtx=this.maskCtx,maskCanvas=this.maskCanvas;imgCtx.clearRect(0,0,this.imgCanvas.width,this.imgCanvas.height),maskCtx.clearRect(0,0,this.maskCanvas.width,this.maskCanvas.height);const filepath=ComfyApp.clipspace.images,alpha_url=new URL(ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex].src);alpha_url.searchParams.delete("channel"),alpha_url.searchParams.delete("preview"),alpha_url.searchParams.set("channel","a");let mask_image=await loadImage(alpha_url);const rgb_url=new URL(ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex].src);rgb_url.searchParams.delete("channel"),rgb_url.searchParams.set("channel","rgb"),this.image=new Image,this.image.onload=function(){maskCanvas.width=self2.image.width,maskCanvas.height=self2.image.height,self2.invalidateCanvas(self2.image,mask_image),self2.initializeCanvasPanZoom()},this.image.src=rgb_url.toString()}initializeCanvasPanZoom(){let drawWidth=this.image.width,drawHeight=this.image.height,width=this.element.clientWidth,height=this.element.clientHeight;this.image.width>width&&(drawWidth=width,drawHeight=drawWidth/this.image.width*this.image.height),drawHeight>height&&(drawHeight=height,drawWidth=drawHeight/this.image.height*this.image.width),this.zoom_ratio=drawWidth/this.image.width;const canvasX=(width-drawWidth)/2,canvasY=(height-drawHeight)/2;this.pan_x=canvasX,this.pan_y=canvasY,this.invalidatePanZoom()}invalidatePanZoom(){let raw_width=this.image.width*this.zoom_ratio,raw_height=this.image.height*this.zoom_ratio;this.pan_x+raw_width<10&&(this.pan_x=10-raw_width),this.pan_y+raw_height<10&&(this.pan_y=10-raw_height);let width=`${raw_width}px`,height=`${raw_height}px`,left=`${this.pan_x}px`,top=`${this.pan_y}px`;this.maskCanvas.style.width=width,this.maskCanvas.style.height=height,this.maskCanvas.style.left=left,this.maskCanvas.style.top=top,this.imgCanvas.style.width=width,this.imgCanvas.style.height=height,this.imgCanvas.style.left=left,this.imgCanvas.style.top=top}setEventHandler(maskCanvas){const self2=this;this.handler_registered||(maskCanvas.addEventListener("contextmenu",event=>{event.preventDefault()}),this.element.addEventListener("wheel",event=>this.handleWheelEvent(self2,event)),this.element.addEventListener("pointermove",event=>this.pointMoveEvent(self2,event)),this.element.addEventListener("touchmove",event=>this.pointMoveEvent(self2,event)),this.element.addEventListener("dragstart",event=>{event.ctrlKey&&event.preventDefault()}),maskCanvas.addEventListener("pointerdown",event=>this.handlePointerDown(self2,event)),maskCanvas.addEventListener("pointermove",event=>this.draw_move(self2,event)),maskCanvas.addEventListener("touchmove",event=>this.draw_move(self2,event)),maskCanvas.addEventListener("pointerover",event=>{this.brush.style.display="block"}),maskCanvas.addEventListener("pointerleave",event=>{this.brush.style.display="none"}),document.addEventListener("pointerup",MaskEditorDialogOld.handlePointerUp),this.handler_registered=!0)}getMaskCanvasStyle(){return this.brush_color_mode==="negative"?{mixBlendMode:"difference",opacity:"1"}:{mixBlendMode:"initial",opacity:this.brush_opacity}}getMaskColor(){return this.brush_color_mode==="black"?{r:0,g:0,b:0}:this.brush_color_mode==="white"?{r:255,g:255,b:255}:this.brush_color_mode==="negative"?{r:255,g:255,b:255}:{r:0,g:0,b:0}}getMaskFillStyle(){const maskColor=this.getMaskColor();return"rgb("+maskColor.r+","+maskColor.g+","+maskColor.b+")"}getColorButtonText(){let colorCaption="unknown";return this.brush_color_mode==="black"?colorCaption="black":this.brush_color_mode==="white"?colorCaption="white":this.brush_color_mode==="negative"&&(colorCaption="negative"),"Color: "+colorCaption}updateWhenBrushColorModeChanged(){this.colorButton.innerText=this.getColorButtonText();const maskCanvasStyle=this.getMaskCanvasStyle();this.maskCanvas.style.mixBlendMode=maskCanvasStyle.mixBlendMode,this.maskCanvas.style.opacity=maskCanvasStyle.opacity.toString();const maskColor=this.getMaskColor(),maskData=this.maskCtx.getImageData(0,0,this.maskCanvas.width,this.maskCanvas.height);for(let i=0;i<maskData.data.length;i+=4)maskData.data[i]=maskColor.r,maskData.data[i+1]=maskColor.g,maskData.data[i+2]=maskColor.b;this.maskCtx.putImageData(maskData,0,0)}brush_opacity=.7;brush_size=10;brush_color_mode="black";drawing_mode=!1;lastx=-1;lasty=-1;lasttime=0;static handleKeyDown(event){const self2=MaskEditorDialogOld.instance;event.key==="]"?(self2.brush_size=Math.min(self2.brush_size+2,100),self2.brush_slider_input.value=self2.brush_size):event.key==="["?(self2.brush_size=Math.max(self2.brush_size-2,1),self2.brush_slider_input.value=self2.brush_size):event.key==="Enter"&&self2.save(),self2.updateBrushPreview(self2)}static handlePointerUp(event){event.preventDefault(),this.mousedown_x=null,this.mousedown_y=null,MaskEditorDialogOld.instance.drawing_mode=!1}updateBrushPreview(self2){const brush=self2.brush;var centerX=self2.cursorX,centerY=self2.cursorY;brush.style.width=self2.brush_size*2*this.zoom_ratio+"px",brush.style.height=self2.brush_size*2*this.zoom_ratio+"px",brush.style.left=centerX-self2.brush_size*this.zoom_ratio+"px",brush.style.top=centerY-self2.brush_size*this.zoom_ratio+"px"}handleWheelEvent(self2,event){event.preventDefault(),event.ctrlKey?(event.deltaY<0?this.zoom_ratio=Math.min(10,this.zoom_ratio+.2):this.zoom_ratio=Math.max(.2,this.zoom_ratio-.2),this.invalidatePanZoom()):(event.deltaY<0?this.brush_size=Math.min(this.brush_size+2,100):this.brush_size=Math.max(this.brush_size-2,1),this.brush_slider_input.value=this.brush_size.toString(),this.updateBrushPreview(this))}pointMoveEvent(self2,event){this.cursorX=event.pageX,this.cursorY=event.pageY,self2.updateBrushPreview(self2),event.ctrlKey&&(event.preventDefault(),self2.pan_move(self2,event));let left_button_down=window.TouchEvent&&event instanceof TouchEvent||event.buttons==1;if(event.shiftKey&&left_button_down){self2.drawing_mode=!1;const y=event.clientY;let delta=(self2.zoom_lasty-y)*.005;self2.zoom_ratio=Math.max(Math.min(10,self2.last_zoom_ratio-delta),.2),this.invalidatePanZoom();return}}pan_move(self2,event){if(event.buttons==1&&MaskEditorDialogOld.mousedown_x){let deltaX=MaskEditorDialogOld.mousedown_x-event.clientX,deltaY=MaskEditorDialogOld.mousedown_y-event.clientY;self2.pan_x=this.mousedown_pan_x-deltaX,self2.pan_y=this.mousedown_pan_y-deltaY,self2.invalidatePanZoom()}}draw_move(self2,event){if(event.ctrlKey||event.shiftKey)return;event.preventDefault(),this.cursorX=event.pageX,this.cursorY=event.pageY,self2.updateBrushPreview(self2);let left_button_down=window.TouchEvent&&event instanceof TouchEvent||event.buttons==1,right_button_down=[2,5,32].includes(event.buttons);if(!event.altKey&&left_button_down){var diff=performance.now()-self2.lasttime;const maskRect=self2.maskCanvas.getBoundingClientRect();var x=event.offsetX,y=event.offsetY;event.offsetX==null&&(x=event.targetTouches[0].clientX-maskRect.left),event.offsetY==null&&(y=event.targetTouches[0].clientY-maskRect.top),x/=self2.zoom_ratio,y/=self2.zoom_ratio;var brush_size=this.brush_size;event instanceof PointerEvent&&event.pointerType=="pen"?(brush_size*=event.pressure,this.last_pressure=event.pressure):window.TouchEvent&&event instanceof TouchEvent&&diff<20?brush_size*=this.last_pressure:brush_size=this.brush_size,diff>20&&!this.drawing_mode?requestAnimationFrame(()=>{self2.init_shape(self2,"source-over"),self2.draw_shape(self2,x,y,brush_size),self2.lastx=x,self2.lasty=y}):requestAnimationFrame(()=>{self2.init_shape(self2,"source-over");for(var dx=x-self2.lastx,dy=y-self2.lasty,distance=Math.sqrt(dx*dx+dy*dy),directionX=dx/distance,directionY=dy/distance,i=0;i<distance;i+=5){var px=self2.lastx+directionX*i,py=self2.lasty+directionY*i;self2.draw_shape(self2,px,py,brush_size)}self2.lastx=x,self2.lasty=y}),self2.lasttime=performance.now()}else if(event.altKey&&left_button_down||right_button_down){const maskRect=self2.maskCanvas.getBoundingClientRect(),x2=(event.offsetX||event.targetTouches[0].clientX-maskRect.left)/self2.zoom_ratio,y2=(event.offsetY||event.targetTouches[0].clientY-maskRect.top)/self2.zoom_ratio;var brush_size=this.brush_size;event instanceof PointerEvent&&event.pointerType=="pen"?(brush_size*=event.pressure,this.last_pressure=event.pressure):window.TouchEvent&&event instanceof TouchEvent&&diff<20?brush_size*=this.last_pressure:brush_size=this.brush_size,diff>20&&!this.drawing_mode?requestAnimationFrame(()=>{self2.init_shape(self2,"destination-out"),self2.draw_shape(self2,x2,y2,brush_size),self2.lastx=x2,self2.lasty=y2}):requestAnimationFrame(()=>{self2.init_shape(self2,"destination-out");for(var dx=x2-self2.lastx,dy=y2-self2.lasty,distance=Math.sqrt(dx*dx+dy*dy),directionX=dx/distance,directionY=dy/distance,i=0;i<distance;i+=5){var px=self2.lastx+directionX*i,py=self2.lasty+directionY*i;self2.draw_shape(self2,px,py,brush_size)}self2.lastx=x2,self2.lasty=y2}),self2.lasttime=performance.now()}}handlePointerDown(self2,event){if(event.ctrlKey){event.buttons==1&&(MaskEditorDialogOld.mousedown_x=event.clientX,MaskEditorDialogOld.mousedown_y=event.clientY,this.mousedown_pan_x=this.pan_x,this.mousedown_pan_y=this.pan_y);return}var brush_size=this.brush_size;if(event instanceof PointerEvent&&event.pointerType=="pen"&&(brush_size*=event.pressure,this.last_pressure=event.pressure),[0,2,5].includes(event.button)){if(self2.drawing_mode=!0,event.preventDefault(),event.shiftKey){self2.zoom_lasty=event.clientY,self2.last_zoom_ratio=self2.zoom_ratio;return}const maskRect=self2.maskCanvas.getBoundingClientRect(),x=(event.offsetX||event.targetTouches[0].clientX-maskRect.left)/self2.zoom_ratio,y=(event.offsetY||event.targetTouches[0].clientY-maskRect.top)/self2.zoom_ratio;!event.altKey&&event.button==0?self2.init_shape(self2,"source-over"):self2.init_shape(self2,"destination-out"),self2.draw_shape(self2,x,y,brush_size),self2.lastx=x,self2.lasty=y,self2.lasttime=performance.now()}}init_shape(self2,compositionOperation){self2.maskCtx.beginPath(),compositionOperation=="source-over"?(self2.maskCtx.fillStyle=this.getMaskFillStyle(),self2.maskCtx.globalCompositeOperation="source-over"):compositionOperation=="destination-out"&&(self2.maskCtx.globalCompositeOperation="destination-out")}draw_shape(self2,x,y,brush_size){self2.pointer_type==="rect"?self2.maskCtx.rect(x-brush_size,y-brush_size,brush_size*2,brush_size*2):self2.maskCtx.arc(x,y,brush_size,0,Math.PI*2,!1),self2.maskCtx.fill()}async save(){const backupCanvas=document.createElement("canvas"),backupCtx=backupCanvas.getContext("2d",{willReadFrequently:!0});backupCanvas.width=this.image.width,backupCanvas.height=this.image.height,backupCtx.clearRect(0,0,backupCanvas.width,backupCanvas.height),backupCtx.drawImage(this.maskCanvas,0,0,this.maskCanvas.width,this.maskCanvas.height,0,0,backupCanvas.width,backupCanvas.height);const backupData=backupCtx.getImageData(0,0,backupCanvas.width,backupCanvas.height);for(let i=0;i<backupData.data.length;i+=4)backupData.data[i+3]==255?backupData.data[i+3]=0:backupData.data[i+3]=255,backupData.data[i]=0,backupData.data[i+1]=0,backupData.data[i+2]=0;backupCtx.globalCompositeOperation="source-over",backupCtx.putImageData(backupData,0,0);const formData=new FormData,filename="clipspace-mask-"+performance.now()+".png",item={filename,subfolder:"clipspace",type:"input"};if(ComfyApp.clipspace.images&&(ComfyApp.clipspace.images[0]=item),ComfyApp.clipspace.widgets){const index=ComfyApp.clipspace.widgets.findIndex(obj=>obj.name==="image");index>=0&&(ComfyApp.clipspace.widgets[index].value=item)}const dataURL=backupCanvas.toDataURL(),blob=dataURLToBlob(dataURL);let original_url=new URL(this.image.src);const original_ref={filename:original_url.searchParams.get("filename")};let original_subfolder=original_url.searchParams.get("subfolder");original_subfolder&&(original_ref.subfolder=original_subfolder);let original_type=original_url.searchParams.get("type");original_type&&(original_ref.type=original_type),formData.append("image",blob,filename),formData.append("original_ref",JSON.stringify(original_ref)),formData.append("type","input"),formData.append("subfolder","clipspace"),this.saveButton.innerText="Saving...",this.saveButton.disabled=!0,await uploadMask(item,formData),ComfyApp.onClipspaceEditorSave(),this.close()}}window.comfyAPI=window.comfyAPI||{};window.comfyAPI.maskEditorOld=window.comfyAPI.maskEditorOld||{};window.comfyAPI.maskEditorOld.MaskEditorDialogOld=MaskEditorDialogOld;var styles=`
  #maskEditorContainer {
    display: fixed;
  }
  #maskEditor_brush {
    position: absolute;
    backgroundColor: transparent;
    z-index: 8889;
    pointer-events: none;
    border-radius: 50%;
    overflow: visible;
    outline: 1px dashed black;
    box-shadow: 0 0 0 1px white;
  }
  #maskEditor_brushPreviewGradient {
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    display: none;
  }
  #maskEditor {
    display: block;
    width: 100%;
    height: 100vh;
    left: 0;
    z-index: 8888;
    position: fixed;
    background: rgba(50,50,50,0.75);
    backdrop-filter: blur(10px);
    overflow: hidden;
    user-select: none;
  }
  #maskEditor_sidePanelContainer {
    height: 100%;
    width: 220px;
    z-index: 8888;
    display: flex;
    flex-direction: column;
  }
  #maskEditor_sidePanel {
    background: var(--comfy-menu-bg);
    height: 100%;
    display: flex;
    align-items: center;
    overflow-y: hidden;
    width: 220px;
  }
  #maskEditor_sidePanelShortcuts {
    display: flex;
    flex-direction: row;
    width: 200px;
    margin-top: 10px;
    gap: 10px;
    justify-content: center;
  }
  .maskEditor_sidePanelIconButton {
    width: 40px;
    height: 40px;
    pointer-events: auto;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: background-color 0.1s;
  }
  .maskEditor_sidePanelIconButton:hover {
    background-color: rgba(0, 0, 0, 0.2);
  }
  #maskEditor_sidePanelBrushSettings {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 200px;
    padding: 10px;
  }
  .maskEditor_sidePanelTitle {
    text-align: center;
    font-size: 15px;
    font-family: sans-serif;
    color: var(--descrip-text);
    margin-top: 10px;
  }
  #maskEditor_sidePanelBrushShapeContainer {
    display: flex;
    width: 180px;
    height: 50px;
    border: 1px solid var(--border-color);
    pointer-events: auto;
    background: rgba(0, 0, 0, 0.2);
  }
  #maskEditor_sidePanelBrushShapeCircle {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    border: 1px solid var(--border-color);
    pointer-events: auto;
    transition: background 0.1s;
    margin-left: 7.5px;
  }
  .maskEditor_sidePanelBrushRange {
    width: 180px;
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    cursor: pointer;
  }
  .maskEditor_sidePanelBrushRange::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    cursor: grab;
    margin-top: -8px;
    background: var(--p-surface-700);
    border: 1px solid var(--border-color);
  }
  .maskEditor_sidePanelBrushRange::-moz-range-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    cursor: grab;
    background: var(--p-surface-800);
    border: 1px solid var(--border-color);
  }
  .maskEditor_sidePanelBrushRange::-webkit-slider-runnable-track {
    background: var(--p-surface-700);
    height: 3px;
  }
  .maskEditor_sidePanelBrushRange::-moz-range-track {
    background: var(--p-surface-700);
    height: 3px;
  }

  #maskEditor_sidePanelBrushShapeSquare {
    width: 35px;
    height: 35px;
    margin: 5px;
    border: 1px solid var(--border-color);
    pointer-events: auto;
    transition: background 0.1s;
  }

  .maskEditor_brushShape_dark {
    background: transparent;
  }

  .maskEditor_brushShape_dark:hover {
    background: var(--p-surface-900);
  }

  .maskEditor_brushShape_light {
    background: transparent;
  }

  .maskEditor_brushShape_light:hover {
    background: var(--comfy-menu-bg);
  }

  #maskEditor_sidePanelImageLayerSettings {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 200px;
    align-items: center;
  }
  .maskEditor_sidePanelLayer {
    display: flex;
    width: 200px;
    height: 50px;
  }
  .maskEditor_sidePanelLayerVisibilityContainer {
    width: 50px;
    height: 50px;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .maskEditor_sidePanelVisibilityToggle {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    pointer-events: auto;
  }
  .maskEditor_sidePanelLayerIconContainer {
    width: 60px;
    height: 50px;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
    fill: var(--input-text);
  }
  .maskEditor_sidePanelLayerIconContainer svg {
    width: 30px;
    height: 30px;
  }
  #maskEditor_sidePanelMaskLayerBlendingContainer {
    width: 80px;
    height: 50px;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  #maskEditor_sidePanelMaskLayerBlendingSelect {
    width: 80px;
    height: 30px;
    border: 1px solid var(--border-color);
    background-color: rgba(0, 0, 0, 0.2);
    color: var(--input-text);
    font-family: sans-serif;
    font-size: 15px;
    pointer-events: auto;
    transition: background-color border 0.1s;
  }
  #maskEditor_sidePanelClearCanvasButton:hover {
    background-color: var(--p-overlaybadge-outline-color);
    border: none;
  }
  #maskEditor_sidePanelClearCanvasButton {
    width: 180px;
    height: 30px;
    border: none;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-color);
    color: var(--input-text);
    font-family: sans-serif;
    font-size: 15px;
    pointer-events: auto;
    transition: background-color 0.1s;
  }
  #maskEditor_sidePanelClearCanvasButton:hover {
    background-color: var(--p-overlaybadge-outline-color);
  }
  #maskEditor_sidePanelHorizontalButtonContainer {
    display: flex;
    gap: 10px;
    height: 40px;
  }
  .maskEditor_sidePanelBigButton {
    width: 85px;
    height: 30px;
    border: none;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border-color);
    color: var(--input-text);
    font-family: sans-serif;
    font-size: 15px;
    pointer-events: auto;
    transition: background-color border 0.1s;
  }
  .maskEditor_sidePanelBigButton:hover {
    background-color: var(--p-overlaybadge-outline-color);
    border: none;
  }
  #maskEditor_toolPanel {
    height: 100%;
    width: var(--sidebar-width);
    z-index: 8888;
    background: var(--comfy-menu-bg);
    display: flex;
    flex-direction: column;
  }
  .maskEditor_toolPanelContainer {
    width: var(--sidebar-width);
    height: var(--sidebar-width);
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    transition: background-color 0.2s;
  }
  .maskEditor_toolPanelContainerSelected svg {
    fill: var(--p-button-text-primary-color) !important;
  }
  .maskEditor_toolPanelContainerSelected .maskEditor_toolPanelIndicator {
    display: block;
  }
  .maskEditor_toolPanelContainer svg {
    width: 75%;
    aspect-ratio: 1/1;
    fill: var(--p-button-text-secondary-color);
  }

  .maskEditor_toolPanelContainerDark:hover {
    background-color: var(--p-surface-800);
  }

  .maskEditor_toolPanelContainerLight:hover {
    background-color: var(--p-surface-300);
  }

  .maskEditor_toolPanelIndicator {
    display: none;
    height: 100%;
    width: 4px;
    position: absolute;
    left: 0;
    background: var(--p-button-text-primary-color);
  }
  #maskEditor_sidePanelPaintBucketSettings {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 200px;
    padding: 10px;
  }
  #canvasBackground {
    background: white;
    width: 100%;
    height: 100%;
  }
  #maskEditor_sidePanelButtonsContainer {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 10px;
  }
  .maskEditor_sidePanelSeparator {
    width: 200px;
    height: 2px;
    background: var(--border-color);
    margin-top: 5px;
    margin-bottom: 5px;
  }
  #maskEditor_pointerZone {
    width: calc(100% - var(--sidebar-width) - 220px);
    height: 100%;
  }
  #maskEditor_uiContainer {
    width: 100%;
    height: 100%;
    position: absolute;
    z-index: 8888;
    display: flex;
    flex-direction: column;
  }
  #maskEditorCanvasContainer {
    position: absolute;
    width: 1000px;
    height: 667px;
    left: 359px;
    top: 280px;
  }
  #imageCanvas {
    width: 100%;
    height: 100%;
  }
  #maskCanvas {
    width: 100%;
    height: 100%;
  }
  #maskEditor_uiHorizontalContainer {
    width: 100%;
    height: 100%;
    display: flex;
  }
  #maskEditor_topBar {
    display: flex;
    height: 44px;
    align-items: center;
    background: var(--comfy-menu-bg);
  }
  #maskEditor_topBarTitle {
    margin: 0;
    margin-left: 0.5rem;
    margin-right: 0.5rem;
    font-size: 1.2em;
  }
  #maskEditor_topBarButtonContainer {
    display: flex;
    gap: 10px;
    margin-right: 0.5rem;
    position: absolute;
    right: 0;
    width: 200px;
  }
  #maskEditor_topBarShortcutsContainer {
    display: flex;
    gap: 10px;
    margin-left: 5px;
  }

  .maskEditor_topPanelIconButton_dark {
    width: 50px;
    height: 30px;
    pointer-events: auto;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: background-color 0.1s;
    background: var(--p-surface-800);
    border: 1px solid var(--p-form-field-border-color);
    border-radius: 10px;
  }

  .maskEditor_topPanelIconButton_dark:hover {
      background-color: var(--p-surface-900);
  }

  .maskEditor_topPanelIconButton_dark svg {
    width: 25px;
    height: 25px;
    pointer-events: none;
    fill: var(--input-text);
  }

  .maskEditor_topPanelIconButton_light {
    width: 50px;
    height: 30px;
    pointer-events: auto;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: background-color 0.1s;
    background: var(--comfy-menu-bg);
    border: 1px solid var(--p-form-field-border-color);
    border-radius: 10px;
  }

  .maskEditor_topPanelIconButton_light:hover {
      background-color: var(--p-surface-300);
  }

  .maskEditor_topPanelIconButton_light svg {
    width: 25px;
    height: 25px;
    pointer-events: none;
    fill: var(--input-text);
  }

  .maskEditor_topPanelButton_dark {
    height: 30px;
    background: var(--p-surface-800);
    border: 1px solid var(--p-form-field-border-color);
    border-radius: 10px;
    color: var(--input-text);
    font-family: sans-serif;
    pointer-events: auto;
    transition: 0.1s;
    width: 60px;
  }

  .maskEditor_topPanelButton_dark:hover {
    background-color: var(--p-surface-900);
  }

  .maskEditor_topPanelButton_light {
    height: 30px;
    background: var(--comfy-menu-bg);
    border: 1px solid var(--p-form-field-border-color);
    border-radius: 10px;
    color: var(--input-text);
    font-family: sans-serif;
    pointer-events: auto;
    transition: 0.1s;
    width: 60px;
  }

  .maskEditor_topPanelButton_light:hover {
    background-color: var(--p-surface-300);
  }


  #maskEditor_sidePanelColorSelectSettings {
    flex-direction: column;
  }
  
  .maskEditor_sidePanel_paintBucket_Container {
    width: 180px;
    display: flex;
    flex-direction: column;
    position: relative;
  }

  .maskEditor_sidePanel_colorSelect_Container {
    display: flex;
    width: 180px;
    align-items: center;
    gap: 5px;
    height: 30px;
  }
  
  #maskEditor_sidePanelVisibilityToggle {
    position: absolute;
    right: 0;
  }

  #maskEditor_sidePanelColorSelectMethodSelect {
    position: absolute;
    right: 0;
    height: 30px;
    border-radius: 0;
    border: 1px solid var(--border-color);
    background: rgba(0,0,0,0.2);
  }

  #maskEditor_sidePanelVisibilityToggle {
    position: absolute;
    right: 0;
  }

  .maskEditor_sidePanel_colorSelect_tolerance_container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 10px;
  }

  .maskEditor_sidePanelContainerColumn {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .maskEditor_sidePanelContainerRow {
    display: flex;
    flex-direction: row;
    gap: 10px;
    align-items: center;
    min-height: 24px;
    position: relative;
  }

  .maskEditor_accent_bg_dark {
    background: var(--p-surface-800);
  }

  .maskEditor_accent_bg_very_dark {
    background: var(--p-surface-900);
  }

  .maskEditor_accent_bg_light {
    background: var(--p-surface-300);
  }

  .maskEditor_accent_bg_very_light {
    background: var(--comfy-menu-bg);
  }

  #maskEditor_paintBucketSettings {
    display: none;
  }

  #maskEditor_colorSelectSettings {
    display: none;
  }

  .maskEditor_sidePanelToggleContainer {
    cursor: pointer;
    display: inline-block;
    position: absolute;
    right: 0;
  }

  .maskEditor_toggle_bg_dark {
    background: var(--p-surface-700);
  }

  .maskEditor_toggle_bg_light {
    background: var(--p-surface-300);
  }

  .maskEditor_sidePanelToggleSwitch {
    display: inline-block;
    border-radius: 16px;
    width: 40px;
    height: 24px;
    position: relative;
    vertical-align: middle;
    transition: background 0.25s;
  }
  .maskEditor_sidePanelToggleSwitch:before, .maskEditor_sidePanelToggleSwitch:after {
    content: "";
  }
  .maskEditor_sidePanelToggleSwitch:before {
    display: block;
    background: linear-gradient(to bottom, #fff 0%, #eee 100%);
    border-radius: 50%;
    width: 16px;
    height: 16px;
    position: absolute;
    top: 4px;
    left: 4px;
    transition: ease 0.2s;
  }
  .maskEditor_sidePanelToggleContainer:hover .maskEditor_sidePanelToggleSwitch:before {
    background: linear-gradient(to bottom, #fff 0%, #fff 100%);
  }
  .maskEditor_sidePanelToggleCheckbox:checked + .maskEditor_sidePanelToggleSwitch {
    background: var(--p-button-text-primary-color);
  }
  .maskEditor_sidePanelToggleCheckbox:checked + .maskEditor_toggle_bg_dark:before {
    background: var(--p-surface-900);
  }
  .maskEditor_sidePanelToggleCheckbox:checked + .maskEditor_toggle_bg_light:before {
    background: var(--comfy-menu-bg);
  }
  .maskEditor_sidePanelToggleCheckbox:checked + .maskEditor_sidePanelToggleSwitch:before {
    left: 20px;
  }

  .maskEditor_sidePanelToggleCheckbox {
    position: absolute;
    visibility: hidden;
  }

  .maskEditor_sidePanelDropdown_dark {
    border: 1px solid var(--p-form-field-border-color);
    background: var(--p-surface-900);
    height: 24px;
    padding-left: 5px;
    padding-right: 5px;
    border-radius: 6px;
    transition: background 0.1s;
  }

  .maskEditor_sidePanelDropdown_dark option {
    background: var(--p-surface-900);
  }

  .maskEditor_sidePanelDropdown_dark:focus {
    outline: 1px solid var(--p-button-text-primary-color);
  }

  .maskEditor_sidePanelDropdown_dark option:hover {
    background: white;
  }
  .maskEditor_sidePanelDropdown_dark option:active {
    background: var(--p-highlight-background);
  }

  .maskEditor_sidePanelDropdown_light {
    border: 1px solid var(--p-form-field-border-color);
    background: var(--comfy-menu-bg);
    height: 24px;
    padding-left: 5px;
    padding-right: 5px;
    border-radius: 6px;
    transition: background 0.1s;
  }

  .maskEditor_sidePanelDropdown_light option {
    background: var(--comfy-menu-bg);
  }

  .maskEditor_sidePanelDropdown_light:focus {
    outline: 1px solid var(--p-surface-300);
  }

  .maskEditor_sidePanelDropdown_light option:hover {
    background: white;
  }
  .maskEditor_sidePanelDropdown_light option:active {
    background: var(--p-surface-300);
  }

  .maskEditor_layerRow {
    height: 50px;
    width: 200px;
    border-radius: 10px;
  }

  .maskEditor_sidePanelLayerPreviewContainer {
    width: 40px;
    height: 30px;
  }

  .maskEditor_sidePanelLayerPreviewContainer > svg{
    width: 100%;
    height: 100%;
    object-fit: contain;
    fill: var(--p-surface-100);
  }

  #maskEditor_sidePanelImageLayerImage {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .maskEditor_sidePanelSubTitle {
    text-align: left;
    font-size: 12px;
    font-family: sans-serif;
    color: var(--descrip-text);
  }

  .maskEditor_containerDropdown {
    position: absolute;
    right: 0;
  }

  .maskEditor_sidePanelLayerCheckbox {
    margin-left: 15px;
  }

  .maskEditor_toolPanelZoomIndicator {
    width: var(--sidebar-width);
    height: var(--sidebar-width);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 5px;
    color: var(--p-button-text-secondary-color);
    position: absolute;
    bottom: 0;
    transition: background-color 0.2s;
  }

  #maskEditor_toolPanelDimensionsText {
    font-size: 12px;
  }

  #maskEditor_topBarSaveButton {
    background: var(--p-primary-color) !important;
    color: var(--p-button-primary-color) !important;
  }

  #maskEditor_topBarSaveButton:hover {
    background: var(--p-primary-hover-color) !important;
  }

`,styleSheet=document.createElement("style");styleSheet.type="text/css";styleSheet.innerText=styles;document.head.appendChild(styleSheet);var ColorComparisonMethod=(ColorComparisonMethod2=>(ColorComparisonMethod2.Simple="simple",ColorComparisonMethod2.HSL="hsl",ColorComparisonMethod2.LAB="lab",ColorComparisonMethod2))(ColorComparisonMethod||{});class MaskEditorDialog extends ComfyDialog{static{__name(this,"MaskEditorDialog")}static instance=null;uiManager;toolManager;panAndZoomManager;brushTool;paintBucketTool;colorSelectTool;canvasHistory;messageBroker;keyboardManager;rootElement;imageURL;isLayoutCreated=!1;isOpen=!1;last_display_style=null;constructor(){super(),this.rootElement=$el("div.maskEditor_hidden",{parent:document.body},[]),this.element=this.rootElement}static getInstance(){if(!ComfyApp.clipspace||!ComfyApp.clipspace.imgs)throw new Error("No clipspace images found");const currentSrc=ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex].src;return(!MaskEditorDialog.instance||currentSrc!==MaskEditorDialog.instance.imageURL)&&(MaskEditorDialog.instance=new MaskEditorDialog),MaskEditorDialog.instance}async show(){if(this.cleanup(),!this.isLayoutCreated){this.messageBroker=new MessageBroker,this.canvasHistory=new CanvasHistory(this,20),this.paintBucketTool=new PaintBucketTool(this),this.brushTool=new BrushTool(this),this.panAndZoomManager=new PanAndZoomManager(this),this.toolManager=new ToolManager(this),this.keyboardManager=new KeyboardManager(this),this.uiManager=new UIManager(this.rootElement,this),this.colorSelectTool=new ColorSelectTool(this);const self2=this,observer=new MutationObserver(function(mutations){mutations.forEach(function(mutation){mutation.type==="attributes"&&mutation.attributeName==="style"&&(self2.last_display_style&&self2.last_display_style!="none"&&self2.element.style.display=="none"&&ComfyApp.onClipspaceEditorClosed(),self2.last_display_style=self2.element.style.display)})}),config={attributes:!0};observer.observe(this.rootElement,config),this.isLayoutCreated=!0,await this.uiManager.setlayout()}this.rootElement.id="maskEditor",this.rootElement.style.display="flex",this.element.style.display="flex",await this.uiManager.initUI(),this.paintBucketTool.initPaintBucketTool(),this.colorSelectTool.initColorSelectTool(),await this.canvasHistory.saveInitialState(),this.isOpen=!0,ComfyApp.clipspace&&ComfyApp.clipspace.imgs&&this.uiManager.setSidebarImage(),this.keyboardManager.addListeners()}cleanup(){document.querySelectorAll('[id^="maskEditor"]').forEach(element=>element.remove()),document.querySelectorAll("#maskEditor_brush").forEach(element=>element.remove())}isOpened(){return this.isOpen}async save(){const backupCanvas=document.createElement("canvas"),imageCanvas=this.uiManager.getImgCanvas(),maskCanvas=this.uiManager.getMaskCanvas(),image=this.uiManager.getImage(),backupCtx=backupCanvas.getContext("2d",{willReadFrequently:!0});if(backupCanvas.width=imageCanvas.width,backupCanvas.height=imageCanvas.height,!backupCtx)return;const maskImageLoaded=new Promise((resolve,reject)=>{const maskImage=new Image;maskImage.src=maskCanvas.toDataURL(),maskImage.onload=()=>{resolve()},maskImage.onerror=error=>{reject(error)}});try{await maskImageLoaded}catch(error){console.error("Error loading mask image:",error);return}backupCtx.clearRect(0,0,backupCanvas.width,backupCanvas.height),backupCtx.drawImage(maskCanvas,0,0,maskCanvas.width,maskCanvas.height,0,0,backupCanvas.width,backupCanvas.height);let maskHasContent=!1;const maskData=backupCtx.getImageData(0,0,backupCanvas.width,backupCanvas.height);for(let i=0;i<maskData.data.length;i+=4)if(maskData.data[i+3]!==0){maskHasContent=!0;break}const backupData=backupCtx.getImageData(0,0,backupCanvas.width,backupCanvas.height);let backupHasContent=!1;for(let i=0;i<backupData.data.length;i+=4)if(backupData.data[i+3]!==0){backupHasContent=!0;break}if(maskHasContent&&!backupHasContent){console.error("Mask appears to be empty"),alert("Cannot save empty mask");return}for(let i=0;i<backupData.data.length;i+=4){const alpha=backupData.data[i+3];backupData.data[i]=0,backupData.data[i+1]=0,backupData.data[i+2]=0,backupData.data[i+3]=255-alpha}backupCtx.globalCompositeOperation="source-over",backupCtx.putImageData(backupData,0,0);const formData=new FormData,filename="clipspace-mask-"+performance.now()+".png",item={filename,subfolder:"clipspace",type:"input"};if(ComfyApp?.clipspace?.widgets?.length){const index=ComfyApp.clipspace.widgets.findIndex(obj=>obj?.name==="image");if(index>=0&&item!==void 0)try{ComfyApp.clipspace.widgets[index].value=item}catch(err2){console.warn("Failed to set widget value:",err2)}}const dataURL=backupCanvas.toDataURL(),blob=this.dataURLToBlob(dataURL);let original_url=new URL(image.src);this.uiManager.setBrushOpacity(0);const filenameRef=original_url.searchParams.get("filename");if(!filenameRef)throw new Error("filename parameter is required");const original_ref={filename:filenameRef};let original_subfolder=original_url.searchParams.get("subfolder");original_subfolder&&(original_ref.subfolder=original_subfolder);let original_type=original_url.searchParams.get("type");original_type&&(original_ref.type=original_type),formData.append("image",blob,filename),formData.append("original_ref",JSON.stringify(original_ref)),formData.append("type","input"),formData.append("subfolder","clipspace"),this.uiManager.setSaveButtonText("Saving"),this.uiManager.setSaveButtonEnabled(!1),this.keyboardManager.removeListeners();const maxRetries=3;let attempt=0,success=!1;for(;attempt<maxRetries&&!success;)try{await this.uploadMask(item,formData),success=!0}catch(error){console.error(`Upload attempt ${attempt+1} failed:`,error),attempt++,attempt<maxRetries?console.log("Retrying upload..."):console.log("Max retries reached. Upload failed.")}success?(ComfyApp.onClipspaceEditorSave(),this.close(),this.isOpen=!1):(this.uiManager.setSaveButtonText("Save"),this.uiManager.setSaveButtonEnabled(!0),this.keyboardManager.addListeners())}getMessageBroker(){return this.messageBroker}dataURLToBlob(dataURL){const parts=dataURL.split(";base64,"),contentType=parts[0].split(":")[1],byteString=atob(parts[1]),arrayBuffer=new ArrayBuffer(byteString.length),uint8Array=new Uint8Array(arrayBuffer);for(let i=0;i<byteString.length;i++)uint8Array[i]=byteString.charCodeAt(i);return new Blob([arrayBuffer],{type:contentType})}async uploadMask(filepath,formData,retries=3){if(retries<=0)throw new Error("Max retries reached");await api.fetchApi("/upload/mask",{method:"POST",body:formData}).then(response=>{response.ok||(console.log("Failed to upload mask:",response),this.uploadMask(filepath,formData,2))}).catch(error=>{console.error("Error:",error)});try{const selectedIndex=ComfyApp.clipspace?.selectedIndex;if(ComfyApp.clipspace?.imgs&&selectedIndex!==void 0){const newImage=new Image;newImage.src=api.apiURL("/view?"+new URLSearchParams(filepath).toString()+app.getPreviewFormatParam()+app.getRandParam()),ComfyApp.clipspace.imgs[selectedIndex]=newImage,ComfyApp.clipspace.images&&(ComfyApp.clipspace.images[selectedIndex]=filepath)}}catch(err2){console.warn("Failed to update clipspace image:",err2)}ClipspaceDialog.invalidatePreview()}}class CanvasHistory{static{__name(this,"CanvasHistory")}maskEditor;messageBroker;canvas;ctx;states=[];currentStateIndex=-1;maxStates=20;initialized=!1;constructor(maskEditor,maxStates=20){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.maxStates=maxStates,this.createListeners()}async pullCanvas(){this.canvas=await this.messageBroker.pull("maskCanvas"),this.ctx=await this.messageBroker.pull("maskCtx")}createListeners(){this.messageBroker.subscribe("saveState",()=>this.saveState()),this.messageBroker.subscribe("undo",()=>this.undo()),this.messageBroker.subscribe("redo",()=>this.redo())}clearStates(){this.states=[],this.currentStateIndex=-1,this.initialized=!1}async saveInitialState(){if(await this.pullCanvas(),!this.canvas.width||!this.canvas.height){requestAnimationFrame(()=>this.saveInitialState());return}this.clearStates();const state=this.ctx.getImageData(0,0,this.canvas.width,this.canvas.height);this.states.push(state),this.currentStateIndex=0,this.initialized=!0}saveState(){if(!this.initialized||this.currentStateIndex===-1){this.saveInitialState();return}this.states=this.states.slice(0,this.currentStateIndex+1);const state=this.ctx.getImageData(0,0,this.canvas.width,this.canvas.height);this.states.push(state),this.currentStateIndex++,this.states.length>this.maxStates&&(this.states.shift(),this.currentStateIndex--)}undo(){this.states.length>1&&this.currentStateIndex>0?(this.currentStateIndex--,this.restoreState(this.states[this.currentStateIndex])):alert("No more undo states available")}redo(){this.states.length>1&&this.currentStateIndex<this.states.length-1?(this.currentStateIndex++,this.restoreState(this.states[this.currentStateIndex])):alert("No more redo states available")}restoreState(state){state&&this.initialized&&this.ctx.putImageData(state,0,0)}}class PaintBucketTool{static{__name(this,"PaintBucketTool")}maskEditor;messageBroker;canvas;ctx;width=null;height=null;imageData=null;data=null;tolerance=5;constructor(maskEditor){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.createListeners(),this.addPullTopics()}initPaintBucketTool(){this.pullCanvas()}async pullCanvas(){this.canvas=await this.messageBroker.pull("maskCanvas"),this.ctx=await this.messageBroker.pull("maskCtx")}createListeners(){this.messageBroker.subscribe("setPaintBucketTolerance",tolerance=>this.setTolerance(tolerance)),this.messageBroker.subscribe("paintBucketFill",point=>this.floodFill(point)),this.messageBroker.subscribe("invert",()=>this.invertMask())}addPullTopics(){this.messageBroker.createPullTopic("getTolerance",async()=>this.tolerance)}getPixel(x,y){return this.data[(y*this.width+x)*4+3]}setPixel(x,y,alpha,color){const index=(y*this.width+x)*4;this.data[index]=color.r,this.data[index+1]=color.g,this.data[index+2]=color.b,this.data[index+3]=alpha}shouldProcessPixel(currentAlpha,targetAlpha,tolerance,isFillMode){return currentAlpha===-1?!1:isFillMode?currentAlpha!==255&&Math.abs(currentAlpha-targetAlpha)<=tolerance:currentAlpha===255||Math.abs(currentAlpha-targetAlpha)<=tolerance}async floodFill(point){let startX=Math.floor(point.x),startY=Math.floor(point.y);if(this.width=this.canvas.width,this.height=this.canvas.height,startX<0||startX>=this.width||startY<0||startY>=this.height)return;this.imageData=this.ctx.getImageData(0,0,this.width,this.height),this.data=this.imageData.data;const targetAlpha=this.getPixel(startX,startY),isFillMode=targetAlpha!==255;if(targetAlpha===-1)return;const maskColor=await this.messageBroker.pull("getMaskColor"),stack=[],visited=new Uint8Array(this.width*this.height);for(this.shouldProcessPixel(targetAlpha,targetAlpha,this.tolerance,isFillMode)&&stack.push([startX,startY]);stack.length>0;){const[x,y]=stack.pop(),visitedIndex=y*this.width+x;if(visited[visitedIndex])continue;const currentAlpha=this.getPixel(x,y);if(!this.shouldProcessPixel(currentAlpha,targetAlpha,this.tolerance,isFillMode))continue;visited[visitedIndex]=1,this.setPixel(x,y,isFillMode?255:0,maskColor);const checkNeighbor=__name((nx,ny)=>{if(!(nx<0||nx>=this.width||ny<0||ny>=this.height)&&!visited[ny*this.width+nx]){const alpha=this.getPixel(nx,ny);this.shouldProcessPixel(alpha,targetAlpha,this.tolerance,isFillMode)&&stack.push([nx,ny])}},"checkNeighbor");checkNeighbor(x-1,y),checkNeighbor(x+1,y),checkNeighbor(x,y-1),checkNeighbor(x,y+1)}this.ctx.putImageData(this.imageData,0,0),this.imageData=null,this.data=null}setTolerance(tolerance){this.tolerance=tolerance}getTolerance(){return this.tolerance}invertMask(){const imageData=this.ctx.getImageData(0,0,this.canvas.width,this.canvas.height),data=imageData.data;let maskR=0,maskG=0,maskB=0;for(let i=0;i<data.length;i+=4)if(data[i+3]>0){maskR=data[i],maskG=data[i+1],maskB=data[i+2];break}for(let i=0;i<data.length;i+=4){const alpha=data[i+3];data[i+3]=255-alpha,alpha===0&&(data[i]=maskR,data[i+1]=maskG,data[i+2]=maskB)}this.ctx.putImageData(imageData,0,0),this.messageBroker.publish("saveState")}}class ColorSelectTool{static{__name(this,"ColorSelectTool")}maskEditor;messageBroker;width=null;height=null;canvas;maskCTX;imageCTX;maskData=null;imageData=null;tolerance=20;livePreview=!1;lastPoint=null;colorComparisonMethod="simple";applyWholeImage=!1;maskBoundry=!1;maskTolerance=0;constructor(maskEditor){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.createListeners(),this.addPullTopics()}async initColorSelectTool(){await this.pullCanvas()}async pullCanvas(){this.canvas=await this.messageBroker.pull("imgCanvas"),this.maskCTX=await this.messageBroker.pull("maskCtx"),this.imageCTX=await this.messageBroker.pull("imageCtx")}createListeners(){this.messageBroker.subscribe("colorSelectFill",point=>this.fillColorSelection(point)),this.messageBroker.subscribe("setColorSelectTolerance",tolerance=>this.setTolerance(tolerance)),this.messageBroker.subscribe("setLivePreview",livePreview=>this.setLivePreview(livePreview)),this.messageBroker.subscribe("setColorComparisonMethod",method=>this.setComparisonMethod(method)),this.messageBroker.subscribe("clearLastPoint",()=>this.clearLastPoint()),this.messageBroker.subscribe("setWholeImage",applyWholeImage=>this.setApplyWholeImage(applyWholeImage)),this.messageBroker.subscribe("setMaskBoundary",maskBoundry=>this.setMaskBoundary(maskBoundry)),this.messageBroker.subscribe("setMaskTolerance",maskTolerance=>this.setMaskTolerance(maskTolerance))}async addPullTopics(){this.messageBroker.createPullTopic("getLivePreview",async()=>this.livePreview)}getPixel(x,y){const index=(y*this.width+x)*4;return{r:this.imageData[index],g:this.imageData[index+1],b:this.imageData[index+2]}}getMaskAlpha(x,y){return this.maskData[(y*this.width+x)*4+3]}isPixelInRange(pixel,target){switch(this.colorComparisonMethod){case"simple":return this.isPixelInRangeSimple(pixel,target);case"hsl":return this.isPixelInRangeHSL(pixel,target);case"lab":return this.isPixelInRangeLab(pixel,target);default:return this.isPixelInRangeSimple(pixel,target)}}isPixelInRangeSimple(pixel,target){return Math.sqrt(Math.pow(pixel.r-target.r,2)+Math.pow(pixel.g-target.g,2)+Math.pow(pixel.b-target.b,2))<=this.tolerance}isPixelInRangeHSL(pixel,target){const pixelHSL=this.rgbToHSL(pixel.r,pixel.g,pixel.b),targetHSL=this.rgbToHSL(target.r,target.g,target.b),hueDiff=Math.abs(pixelHSL.h-targetHSL.h),satDiff=Math.abs(pixelHSL.s-targetHSL.s),lightDiff=Math.abs(pixelHSL.l-targetHSL.l);return Math.sqrt(Math.pow(hueDiff/360*255,2)+Math.pow(satDiff/100*255,2)+Math.pow(lightDiff/100*255,2))<=this.tolerance}rgbToHSL(r,g,b){r/=255,g/=255,b/=255;const max2=Math.max(r,g,b),min=Math.min(r,g,b);let h=0,s=0,l=(max2+min)/2;if(max2!==min){const d=max2-min;switch(s=l>.5?d/(2-max2-min):d/(max2+min),max2){case r:h=(g-b)/d+(g<b?6:0);break;case g:h=(b-r)/d+2;break;case b:h=(r-g)/d+4;break}h/=6}return{h:h*360,s:s*100,l:l*100}}isPixelInRangeLab(pixel,target){const pixelLab=this.rgbToLab(pixel),targetLab=this.rgbToLab(target);return Math.sqrt(Math.pow(pixelLab.l-targetLab.l,2)+Math.pow(pixelLab.a-targetLab.a,2)+Math.pow(pixelLab.b-targetLab.b,2))/100*255<=this.tolerance}rgbToLab(rgb){let r=rgb.r/255,g=rgb.g/255,b=rgb.b/255;r=r>.04045?Math.pow((r+.055)/1.055,2.4):r/12.92,g=g>.04045?Math.pow((g+.055)/1.055,2.4):g/12.92,b=b>.04045?Math.pow((b+.055)/1.055,2.4):b/12.92,r*=100,g*=100,b*=100;const x=r*.4124+g*.3576+b*.1805,y=r*.2126+g*.7152+b*.0722,z=r*.0193+g*.1192+b*.9505,xyz=[x/95.047,y/100,z/108.883];for(let i=0;i<xyz.length;i++)xyz[i]=xyz[i]>.008856?Math.pow(xyz[i],1/3):7.787*xyz[i]+16/116;return{l:116*xyz[1]-16,a:500*(xyz[0]-xyz[1]),b:200*(xyz[1]-xyz[2])}}setPixel(x,y,alpha,color){const index=(y*this.width+x)*4;this.maskData[index]=color.r,this.maskData[index+1]=color.g,this.maskData[index+2]=color.b,this.maskData[index+3]=alpha}async fillColorSelection(point){this.width=this.canvas.width,this.height=this.canvas.height,this.lastPoint=point;const maskData=this.maskCTX.getImageData(0,0,this.width,this.height);if(this.maskData=maskData.data,this.imageData=this.imageCTX.getImageData(0,0,this.width,this.height).data,this.applyWholeImage){const targetPixel=this.getPixel(Math.floor(point.x),Math.floor(point.y)),maskColor=await this.messageBroker.pull("getMaskColor"),width=this.width,height=this.height,CHUNK_SIZE=1e4;for(let i=0;i<width*height;i+=CHUNK_SIZE){const endIndex=Math.min(i+CHUNK_SIZE,width*height);for(let pixelIndex=i;pixelIndex<endIndex;pixelIndex++){const x=pixelIndex%width,y=Math.floor(pixelIndex/width);this.isPixelInRange(this.getPixel(x,y),targetPixel)&&this.setPixel(x,y,255,maskColor)}await new Promise(resolve=>setTimeout(resolve,0))}}else{let startX=Math.floor(point.x),startY=Math.floor(point.y);if(startX<0||startX>=this.width||startY<0||startY>=this.height)return;const pixel=this.getPixel(startX,startY),stack=[],visited=new Uint8Array(this.width*this.height);stack.push([startX,startY]);const maskColor=await this.messageBroker.pull("getMaskColor");for(;stack.length>0;){const[x,y]=stack.pop(),visitedIndex=y*this.width+x;visited[visitedIndex]||!this.isPixelInRange(this.getPixel(x,y),pixel)||(visited[visitedIndex]=1,this.setPixel(x,y,255,maskColor),x>0&&!visited[y*this.width+(x-1)]&&this.isPixelInRange(this.getPixel(x-1,y),pixel)&&(!this.maskBoundry||255-this.getMaskAlpha(x-1,y)>this.maskTolerance)&&stack.push([x-1,y]),x<this.width-1&&!visited[y*this.width+(x+1)]&&this.isPixelInRange(this.getPixel(x+1,y),pixel)&&(!this.maskBoundry||255-this.getMaskAlpha(x+1,y)>this.maskTolerance)&&stack.push([x+1,y]),y>0&&!visited[(y-1)*this.width+x]&&this.isPixelInRange(this.getPixel(x,y-1),pixel)&&(!this.maskBoundry||255-this.getMaskAlpha(x,y-1)>this.maskTolerance)&&stack.push([x,y-1]),y<this.height-1&&!visited[(y+1)*this.width+x]&&this.isPixelInRange(this.getPixel(x,y+1),pixel)&&(!this.maskBoundry||255-this.getMaskAlpha(x,y+1)>this.maskTolerance)&&stack.push([x,y+1]))}}this.maskCTX.putImageData(maskData,0,0),this.messageBroker.publish("saveState"),this.maskData=null,this.imageData=null}setTolerance(tolerance){this.tolerance=tolerance,this.lastPoint&&this.livePreview&&(this.messageBroker.publish("undo"),this.fillColorSelection(this.lastPoint))}setLivePreview(livePreview){this.livePreview=livePreview}setComparisonMethod(method){this.colorComparisonMethod=method,this.lastPoint&&this.livePreview&&(this.messageBroker.publish("undo"),this.fillColorSelection(this.lastPoint))}clearLastPoint(){this.lastPoint=null}setApplyWholeImage(applyWholeImage){this.applyWholeImage=applyWholeImage}setMaskBoundary(maskBoundry){this.maskBoundry=maskBoundry}setMaskTolerance(maskTolerance){this.maskTolerance=maskTolerance}}class BrushTool{static{__name(this,"BrushTool")}brushSettings;maskBlendMode;isDrawing=!1;isDrawingLine=!1;lineStartPoint=null;smoothingPrecision=10;smoothingCordsArray=[];smoothingLastDrawTime;maskCtx=null;initialDraw=!0;brushStrokeCanvas=null;brushStrokeCtx=null;isBrushAdjusting=!1;brushPreviewGradient=null;initialPoint=null;useDominantAxis=!1;brushAdjustmentSpeed=1;maskEditor;messageBroker;constructor(maskEditor){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.createListeners(),this.addPullTopics(),this.useDominantAxis=app.extensionManager.setting.get("Comfy.MaskEditor.UseDominantAxis"),this.brushAdjustmentSpeed=app.extensionManager.setting.get("Comfy.MaskEditor.BrushAdjustmentSpeed"),this.brushSettings={size:10,opacity:100,hardness:1,type:"arc"},this.maskBlendMode="black"}createListeners(){this.messageBroker.subscribe("setBrushSize",size=>this.setBrushSize(size)),this.messageBroker.subscribe("setBrushOpacity",opacity=>this.setBrushOpacity(opacity)),this.messageBroker.subscribe("setBrushHardness",hardness=>this.setBrushHardness(hardness)),this.messageBroker.subscribe("setBrushShape",type=>this.setBrushType(type)),this.messageBroker.subscribe("setBrushSmoothingPrecision",precision=>this.setBrushSmoothingPrecision(precision)),this.messageBroker.subscribe("brushAdjustmentStart",event=>this.startBrushAdjustment(event)),this.messageBroker.subscribe("brushAdjustment",event=>this.handleBrushAdjustment(event)),this.messageBroker.subscribe("drawStart",event=>this.startDrawing(event)),this.messageBroker.subscribe("draw",event=>this.handleDrawing(event)),this.messageBroker.subscribe("drawEnd",event=>this.drawEnd(event))}addPullTopics(){this.messageBroker.createPullTopic("brushSize",async()=>this.brushSettings.size),this.messageBroker.createPullTopic("brushOpacity",async()=>this.brushSettings.opacity),this.messageBroker.createPullTopic("brushHardness",async()=>this.brushSettings.hardness),this.messageBroker.createPullTopic("brushType",async()=>this.brushSettings.type),this.messageBroker.createPullTopic("maskBlendMode",async()=>this.maskBlendMode),this.messageBroker.createPullTopic("brushSettings",async()=>this.brushSettings)}async createBrushStrokeCanvas(){if(this.brushStrokeCanvas!==null)return;const maskCanvas=await this.messageBroker.pull("maskCanvas"),canvas=document.createElement("canvas");canvas.width=maskCanvas.width,canvas.height=maskCanvas.height,this.brushStrokeCanvas=canvas,this.brushStrokeCtx=canvas.getContext("2d")}async startDrawing(event){this.isDrawing=!0;let compositionOp,currentTool=await this.messageBroker.pull("currentTool"),coords={x:event.offsetX,y:event.offsetY},coords_canvas=await this.messageBroker.pull("screenToCanvas",coords);await this.createBrushStrokeCanvas(),currentTool==="eraser"||event.buttons==2?compositionOp="destination-out":compositionOp="source-over",event.shiftKey&&this.lineStartPoint?(this.isDrawingLine=!0,this.drawLine(this.lineStartPoint,coords_canvas,compositionOp)):(this.isDrawingLine=!1,this.init_shape(compositionOp),this.draw_shape(coords_canvas)),this.lineStartPoint=coords_canvas,this.smoothingCordsArray=[coords_canvas],this.smoothingLastDrawTime=new Date}async handleDrawing(event){var diff=performance.now()-this.smoothingLastDrawTime.getTime();let coords={x:event.offsetX,y:event.offsetY},coords_canvas=await this.messageBroker.pull("screenToCanvas",coords),currentTool=await this.messageBroker.pull("currentTool");diff>20&&!this.isDrawing?requestAnimationFrame(()=>{this.init_shape("source-over"),this.draw_shape(coords_canvas),this.smoothingCordsArray.push(coords_canvas)}):requestAnimationFrame(()=>{currentTool==="eraser"||event.buttons==2?this.init_shape("destination-out"):this.init_shape("source-over"),this.drawWithBetterSmoothing(coords_canvas)}),this.smoothingLastDrawTime=new Date}async drawEnd(event){const coords={x:event.offsetX,y:event.offsetY},coords_canvas=await this.messageBroker.pull("screenToCanvas",coords);this.isDrawing&&(this.isDrawing=!1,this.messageBroker.publish("saveState"),this.lineStartPoint=coords_canvas,this.initialDraw=!0)}drawWithBetterSmoothing(point){this.smoothingCordsArray||(this.smoothingCordsArray=[]);const opacityConstant=1/(1+Math.exp(3)),interpolatedOpacity=1/(1+Math.exp(-6*(this.brushSettings.opacity-.5)))-opacityConstant;if(this.smoothingCordsArray.push(point),this.smoothingCordsArray.length<5)return;let totalLength=0;const points=this.smoothingCordsArray,len=points.length-1;let dx,dy;for(let i=0;i<len;i++)dx=points[i+1].x-points[i].x,dy=points[i+1].y-points[i].y,totalLength+=Math.sqrt(dx*dx+dy*dy);const distanceBetweenPoints=this.brushSettings.size/this.smoothingPrecision*6,stepNr=Math.ceil(totalLength/distanceBetweenPoints);let interpolatedPoints=points;if(stepNr>0&&(interpolatedPoints=this.generateEquidistantPoints(this.smoothingCordsArray,distanceBetweenPoints)),!this.initialDraw){const spliceIndex=interpolatedPoints.findIndex(point2=>point2.x===this.smoothingCordsArray[2].x&&point2.y===this.smoothingCordsArray[2].y);spliceIndex!==-1&&(interpolatedPoints=interpolatedPoints.slice(spliceIndex+1))}for(const point2 of interpolatedPoints)this.draw_shape(point2,interpolatedOpacity);this.initialDraw?this.initialDraw=!1:this.smoothingCordsArray=this.smoothingCordsArray.slice(2)}async drawLine(p1,p2,compositionOp){const brush_size=await this.messageBroker.pull("brushSize"),distance=Math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2),steps=Math.ceil(distance/(brush_size/this.smoothingPrecision*4)),interpolatedOpacity=1/(1+Math.exp(-6*(this.brushSettings.opacity-.5)))-1/(1+Math.exp(3));this.init_shape(compositionOp);for(let i=0;i<=steps;i++){const t2=i/steps,x=p1.x+(p2.x-p1.x)*t2,y=p1.y+(p2.y-p1.y)*t2,point={x,y};this.draw_shape(point,interpolatedOpacity)}}async startBrushAdjustment(event){event.preventDefault();const coords={x:event.offsetX,y:event.offsetY};let coords_canvas=await this.messageBroker.pull("screenToCanvas",coords);this.messageBroker.publish("setBrushPreviewGradientVisibility",!0),this.initialPoint=coords_canvas,this.isBrushAdjusting=!0}async handleBrushAdjustment(event){const coords={x:event.offsetX,y:event.offsetY},brushDeadZone=5;let coords_canvas=await this.messageBroker.pull("screenToCanvas",coords);const delta_x=coords_canvas.x-this.initialPoint.x,delta_y=coords_canvas.y-this.initialPoint.y,effectiveDeltaX=Math.abs(delta_x)<brushDeadZone?0:delta_x,effectiveDeltaY=Math.abs(delta_y)<brushDeadZone?0:delta_y;let finalDeltaX=effectiveDeltaX,finalDeltaY=effectiveDeltaY;if(console.log(this.useDominantAxis),this.useDominantAxis){const ratio=Math.abs(effectiveDeltaX)/Math.abs(effectiveDeltaY),threshold=2;ratio>threshold?finalDeltaY=0:ratio<1/threshold&&(finalDeltaX=0)}const cappedDeltaX=Math.max(-100,Math.min(100,finalDeltaX)),cappedDeltaY=Math.max(-100,Math.min(100,finalDeltaY)),sizeDelta=cappedDeltaX/40,hardnessDelta=cappedDeltaY/800,newSize=Math.max(1,Math.min(100,this.brushSettings.size+cappedDeltaX/35*this.brushAdjustmentSpeed)),newHardness=Math.max(0,Math.min(1,this.brushSettings.hardness-cappedDeltaY/4e3*this.brushAdjustmentSpeed));this.brushSettings.size=newSize,this.brushSettings.hardness=newHardness,this.messageBroker.publish("updateBrushPreview")}async draw_shape(point,overrideOpacity){const brushSettings=this.brushSettings,maskCtx=this.maskCtx||await this.messageBroker.pull("maskCtx"),brushType=await this.messageBroker.pull("brushType"),maskColor=await this.messageBroker.pull("getMaskColor"),size=brushSettings.size,sliderOpacity=brushSettings.opacity,opacity=overrideOpacity??sliderOpacity,hardness=brushSettings.hardness,x=point.x,y=point.y,extendedSize=size*(2-hardness);let gradient=maskCtx.createRadialGradient(x,y,0,x,y,extendedSize);const isErasing=maskCtx.globalCompositeOperation==="destination-out";if(hardness===1)console.log(sliderOpacity,opacity),gradient.addColorStop(0,isErasing?`rgba(255, 255, 255, ${opacity})`:`rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity})`),gradient.addColorStop(1,isErasing?`rgba(255, 255, 255, ${opacity})`:`rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity})`);else{let softness=1-hardness,innerStop=Math.max(0,hardness-softness),outerStop=size/extendedSize;isErasing?(gradient.addColorStop(0,`rgba(255, 255, 255, ${opacity})`),gradient.addColorStop(innerStop,`rgba(255, 255, 255, ${opacity})`),gradient.addColorStop(outerStop,`rgba(255, 255, 255, ${opacity/2})`),gradient.addColorStop(1,"rgba(255, 255, 255, 0)")):(gradient.addColorStop(0,`rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity})`),gradient.addColorStop(innerStop,`rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity})`),gradient.addColorStop(outerStop,`rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity/2})`),gradient.addColorStop(1,`rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, 0)`))}maskCtx.fillStyle=gradient,maskCtx.beginPath(),brushType==="rect"?maskCtx.rect(x-extendedSize,y-extendedSize,extendedSize*2,extendedSize*2):maskCtx.arc(x,y,extendedSize,0,Math.PI*2,!1),maskCtx.fill()}async init_shape(compositionOperation){const maskBlendMode=await this.messageBroker.pull("maskBlendMode"),maskCtx=this.maskCtx||await this.messageBroker.pull("maskCtx");maskCtx.beginPath(),compositionOperation=="source-over"?(maskCtx.fillStyle=maskBlendMode,maskCtx.globalCompositeOperation="source-over"):compositionOperation=="destination-out"&&(maskCtx.globalCompositeOperation="destination-out")}calculateCubicSplinePoints(points,numSegments=10){const result=[],xCoords=points.map(p=>p.x),yCoords=points.map(p=>p.y),xDerivatives=this.calculateSplineCoefficients(xCoords),yDerivatives=this.calculateSplineCoefficients(yCoords);for(let i=0;i<points.length-1;i++){const p0=points[i],p1=points[i+1],d0x=xDerivatives[i],d1x=xDerivatives[i+1],d0y=yDerivatives[i],d1y=yDerivatives[i+1];for(let t2=0;t2<=numSegments;t2++){const t_normalized=t2/numSegments,h00=2*t_normalized**3-3*t_normalized**2+1,h10=t_normalized**3-2*t_normalized**2+t_normalized,h01=-2*t_normalized**3+3*t_normalized**2,h11=t_normalized**3-t_normalized**2,x=h00*p0.x+h10*d0x+h01*p1.x+h11*d1x,y=h00*p0.y+h10*d0y+h01*p1.y+h11*d1y;result.push({x,y})}}return result}generateEvenlyDistributedPoints(splinePoints,numPoints){const distances=[0];for(let i=1;i<splinePoints.length;i++){const dx=splinePoints[i].x-splinePoints[i-1].x,dy=splinePoints[i].y-splinePoints[i-1].y,dist=Math.hypot(dx,dy);distances.push(distances[i-1]+dist)}const interval=distances[distances.length-1]/(numPoints-1),result=[];let currentIndex=0;for(let i=0;i<numPoints;i++){const targetDistance=i*interval;for(;currentIndex<distances.length-1&&distances[currentIndex+1]<targetDistance;)currentIndex++;const t2=(targetDistance-distances[currentIndex])/(distances[currentIndex+1]-distances[currentIndex]),x=splinePoints[currentIndex].x+t2*(splinePoints[currentIndex+1].x-splinePoints[currentIndex].x),y=splinePoints[currentIndex].y+t2*(splinePoints[currentIndex+1].y-splinePoints[currentIndex].y);result.push({x,y})}return result}generateEquidistantPoints(points,distance){const result=[],cumulativeDistances=[0];for(let i=1;i<points.length;i++){const dx=points[i].x-points[i-1].x,dy=points[i].y-points[i-1].y,dist=Math.hypot(dx,dy);cumulativeDistances[i]=cumulativeDistances[i-1]+dist}const totalLength=cumulativeDistances[cumulativeDistances.length-1],numPoints=Math.floor(totalLength/distance);for(let i=0;i<=numPoints;i++){const targetDistance=i*distance;let idx=0;for(;idx<cumulativeDistances.length-1&&cumulativeDistances[idx+1]<targetDistance;)idx++;if(idx>=points.length-1){result.push(points[points.length-1]);continue}const d0=cumulativeDistances[idx],d1=cumulativeDistances[idx+1],t2=(targetDistance-d0)/(d1-d0),x=points[idx].x+t2*(points[idx+1].x-points[idx].x),y=points[idx].y+t2*(points[idx+1].y-points[idx].y);result.push({x,y})}return result}calculateSplineCoefficients(values){const n=values.length-1,matrix=new Array(n+1).fill(0).map(()=>new Array(n+1).fill(0)),rhs=new Array(n+1).fill(0);for(let i=1;i<n;i++)matrix[i][i-1]=1,matrix[i][i]=4,matrix[i][i+1]=1,rhs[i]=3*(values[i+1]-values[i-1]);matrix[0][0]=2,matrix[0][1]=1,matrix[n][n-1]=1,matrix[n][n]=2,rhs[0]=3*(values[1]-values[0]),rhs[n]=3*(values[n]-values[n-1]);for(let i=1;i<=n;i++){const m=matrix[i][i-1]/matrix[i-1][i-1];matrix[i][i]-=m*matrix[i-1][i],rhs[i]-=m*rhs[i-1]}const solution=new Array(n+1);solution[n]=rhs[n]/matrix[n][n];for(let i=n-1;i>=0;i--)solution[i]=(rhs[i]-matrix[i][i+1]*solution[i+1])/matrix[i][i];return solution}setBrushSize(size){this.brushSettings.size=size}setBrushOpacity(opacity){this.brushSettings.opacity=opacity}setBrushHardness(hardness){this.brushSettings.hardness=hardness}setBrushType(type){this.brushSettings.type=type}setBrushSmoothingPrecision(precision){this.smoothingPrecision=precision}}class UIManager{static{__name(this,"UIManager")}rootElement;brush;brushPreviewGradient;maskCtx;imageCtx;maskCanvas;imgCanvas;brushSettingsHTML;paintBucketSettingsHTML;colorSelectSettingsHTML;maskOpacitySlider;brushHardnessSlider;brushSizeSlider;brushOpacitySlider;sidebarImage;saveButton;toolPanel;sidePanel;pointerZone;canvasBackground;canvasContainer;image;imageURL;darkMode=!0;maskEditor;messageBroker;mask_opacity=1;maskBlendMode="black";zoomTextHTML;dimensionsTextHTML;constructor(rootElement,maskEditor){this.rootElement=rootElement,this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.addListeners(),this.addPullTopics()}addListeners(){this.messageBroker.subscribe("updateBrushPreview",async()=>this.updateBrushPreview()),this.messageBroker.subscribe("paintBucketCursor",isPaintBucket=>this.handlePaintBucketCursor(isPaintBucket)),this.messageBroker.subscribe("panCursor",isPan=>this.handlePanCursor(isPan)),this.messageBroker.subscribe("setBrushVisibility",isVisible=>this.setBrushVisibility(isVisible)),this.messageBroker.subscribe("setBrushPreviewGradientVisibility",isVisible=>this.setBrushPreviewGradientVisibility(isVisible)),this.messageBroker.subscribe("updateCursor",()=>this.updateCursor()),this.messageBroker.subscribe("setZoomText",text=>this.setZoomText(text))}addPullTopics(){this.messageBroker.createPullTopic("maskCanvas",async()=>this.maskCanvas),this.messageBroker.createPullTopic("maskCtx",async()=>this.maskCtx),this.messageBroker.createPullTopic("imageCtx",async()=>this.imageCtx),this.messageBroker.createPullTopic("imgCanvas",async()=>this.imgCanvas),this.messageBroker.createPullTopic("screenToCanvas",async coords=>this.screenToCanvas(coords)),this.messageBroker.createPullTopic("getCanvasContainer",async()=>this.canvasContainer),this.messageBroker.createPullTopic("getMaskColor",async()=>this.getMaskColor())}async setlayout(){this.detectLightMode();var user_ui=await this.createUI(),canvasContainer=this.createBackgroundUI(),brush=await this.createBrush();await this.setBrushBorderRadius(),this.setBrushOpacity(1),this.rootElement.appendChild(canvasContainer),this.rootElement.appendChild(user_ui),document.body.appendChild(brush)}async createUI(){var ui_container=document.createElement("div");ui_container.id="maskEditor_uiContainer";var top_bar=await this.createTopBar(),ui_horizontal_container=document.createElement("div");ui_horizontal_container.id="maskEditor_uiHorizontalContainer";var side_panel_container=await this.createSidePanel(),pointer_zone=this.createPointerZone(),tool_panel=this.createToolPanel();return ui_horizontal_container.appendChild(tool_panel),ui_horizontal_container.appendChild(pointer_zone),ui_horizontal_container.appendChild(side_panel_container),ui_container.appendChild(top_bar),ui_container.appendChild(ui_horizontal_container),ui_container}createBackgroundUI(){const canvasContainer=document.createElement("div");canvasContainer.id="maskEditorCanvasContainer";const imgCanvas=document.createElement("canvas");imgCanvas.id="imageCanvas";const maskCanvas=document.createElement("canvas");maskCanvas.id="maskCanvas";const canvas_background=document.createElement("div");canvas_background.id="canvasBackground",canvasContainer.appendChild(imgCanvas),canvasContainer.appendChild(maskCanvas),canvasContainer.appendChild(canvas_background),this.imgCanvas=imgCanvas,this.maskCanvas=maskCanvas,this.canvasContainer=canvasContainer,this.canvasBackground=canvas_background;let maskCtx=maskCanvas.getContext("2d",{willReadFrequently:!0});maskCtx&&(this.maskCtx=maskCtx);let imgCtx=imgCanvas.getContext("2d",{willReadFrequently:!0});imgCtx&&(this.imageCtx=imgCtx),this.setEventHandler(),this.imgCanvas.style.position="absolute",this.maskCanvas.style.position="absolute",this.imgCanvas.style.top="200",this.imgCanvas.style.left="0",this.maskCanvas.style.top=this.imgCanvas.style.top,this.maskCanvas.style.left=this.imgCanvas.style.left;const maskCanvasStyle=this.getMaskCanvasStyle();return this.maskCanvas.style.mixBlendMode=maskCanvasStyle.mixBlendMode,this.maskCanvas.style.opacity=maskCanvasStyle.opacity.toString(),canvasContainer}async setBrushBorderRadius(){(await this.messageBroker.pull("brushSettings")).type==="rect"?(this.brush.style.borderRadius="0%",this.brush.style.MozBorderRadius="0%",this.brush.style.WebkitBorderRadius="0%"):(this.brush.style.borderRadius="50%",this.brush.style.MozBorderRadius="50%",this.brush.style.WebkitBorderRadius="50%")}async initUI(){this.saveButton.innerText="Save",this.saveButton.disabled=!1,await this.setImages(this.imgCanvas)}async createSidePanel(){const side_panel=this.createContainer(!0);side_panel.id="maskEditor_sidePanel";const brush_settings=await this.createBrushSettings();brush_settings.id="maskEditor_brushSettings",this.brushSettingsHTML=brush_settings;const paint_bucket_settings=await this.createPaintBucketSettings();paint_bucket_settings.id="maskEditor_paintBucketSettings",this.paintBucketSettingsHTML=paint_bucket_settings;const color_select_settings=await this.createColorSelectSettings();color_select_settings.id="maskEditor_colorSelectSettings",this.colorSelectSettingsHTML=color_select_settings;const image_layer_settings=await this.createImageLayerSettings(),separator=this.createSeparator();return side_panel.appendChild(brush_settings),side_panel.appendChild(paint_bucket_settings),side_panel.appendChild(color_select_settings),side_panel.appendChild(separator),side_panel.appendChild(image_layer_settings),side_panel}async createBrushSettings(){const shapeColor=this.darkMode?"maskEditor_brushShape_dark":"maskEditor_brushShape_light",brush_settings_container=this.createContainer(!0),brush_settings_title=this.createHeadline("Brush Settings"),brush_shape_outer_container=this.createContainer(!0),brush_shape_title=this.createContainerTitle("Brush Shape"),brush_shape_container=this.createContainer(!1),accentColor=this.darkMode?"maskEditor_accent_bg_dark":"maskEditor_accent_bg_light";brush_shape_container.classList.add(accentColor),brush_shape_container.classList.add("maskEditor_layerRow");const circle_shape=document.createElement("div");circle_shape.id="maskEditor_sidePanelBrushShapeCircle",circle_shape.classList.add(shapeColor),circle_shape.style.background="var(--p-button-text-primary-color)",circle_shape.addEventListener("click",()=>{this.messageBroker.publish("setBrushShape","arc"),this.setBrushBorderRadius(),circle_shape.style.background="var(--p-button-text-primary-color)",square_shape.style.background=""});const square_shape=document.createElement("div");square_shape.id="maskEditor_sidePanelBrushShapeSquare",square_shape.classList.add(shapeColor),square_shape.style.background="",square_shape.addEventListener("click",()=>{this.messageBroker.publish("setBrushShape","rect"),this.setBrushBorderRadius(),square_shape.style.background="var(--p-button-text-primary-color)",circle_shape.style.background=""}),brush_shape_container.appendChild(circle_shape),brush_shape_container.appendChild(square_shape),brush_shape_outer_container.appendChild(brush_shape_title),brush_shape_outer_container.appendChild(brush_shape_container);const thicknesSliderObj=this.createSlider("Thickness",1,100,1,10,(event,value)=>{this.messageBroker.publish("setBrushSize",parseInt(value)),this.updateBrushPreview()});this.brushSizeSlider=thicknesSliderObj.slider;const opacitySliderObj=this.createSlider("Opacity",0,1,.01,.7,(event,value)=>{this.messageBroker.publish("setBrushOpacity",parseFloat(value)),this.updateBrushPreview()});this.brushOpacitySlider=opacitySliderObj.slider;const hardnessSliderObj=this.createSlider("Hardness",0,1,.01,1,(event,value)=>{this.messageBroker.publish("setBrushHardness",parseFloat(value)),this.updateBrushPreview()});this.brushHardnessSlider=hardnessSliderObj.slider;const brushSmoothingPrecisionSliderObj=this.createSlider("Smoothing Precision",1,100,1,10,(event,value)=>{this.messageBroker.publish("setBrushSmoothingPrecision",parseInt(value))});return brush_settings_container.appendChild(brush_settings_title),brush_settings_container.appendChild(brush_shape_outer_container),brush_settings_container.appendChild(thicknesSliderObj.container),brush_settings_container.appendChild(opacitySliderObj.container),brush_settings_container.appendChild(hardnessSliderObj.container),brush_settings_container.appendChild(brushSmoothingPrecisionSliderObj.container),brush_settings_container}async createPaintBucketSettings(){const paint_bucket_settings_container=this.createContainer(!0),paint_bucket_settings_title=this.createHeadline("Paint Bucket Settings"),tolerance=await this.messageBroker.pull("getTolerance"),paintBucketToleranceSliderObj=this.createSlider("Tolerance",0,255,1,tolerance,(event,value)=>{this.messageBroker.publish("setPaintBucketTolerance",parseInt(value))});return paint_bucket_settings_container.appendChild(paint_bucket_settings_title),paint_bucket_settings_container.appendChild(paintBucketToleranceSliderObj.container),paint_bucket_settings_container}async createColorSelectSettings(){const color_select_settings_container=this.createContainer(!0),color_select_settings_title=this.createHeadline("Color Select Settings");var tolerance=await this.messageBroker.pull("getTolerance");const colorSelectToleranceSliderObj=this.createSlider("Tolerance",0,255,1,tolerance,(event,value)=>{this.messageBroker.publish("setColorSelectTolerance",parseInt(value))}),livePreviewToggle=this.createToggle("Live Preview",(event,value)=>{this.messageBroker.publish("setLivePreview",value)}),wholeImageToggle=this.createToggle("Apply to Whole Image",(event,value)=>{this.messageBroker.publish("setWholeImage",value)}),methodOptions=Object.values(ColorComparisonMethod),methodSelect=this.createDropdown("Method",methodOptions,(event,value)=>{this.messageBroker.publish("setColorComparisonMethod",value)}),maskBoundaryToggle=this.createToggle("Stop at mask",(event,value)=>{this.messageBroker.publish("setMaskBoundary",value)}),maskToleranceSliderObj=this.createSlider("Mask Tolerance",0,255,1,0,(event,value)=>{this.messageBroker.publish("setMaskTolerance",parseInt(value))});return color_select_settings_container.appendChild(color_select_settings_title),color_select_settings_container.appendChild(colorSelectToleranceSliderObj.container),color_select_settings_container.appendChild(livePreviewToggle),color_select_settings_container.appendChild(wholeImageToggle),color_select_settings_container.appendChild(methodSelect),color_select_settings_container.appendChild(maskBoundaryToggle),color_select_settings_container.appendChild(maskToleranceSliderObj.container),color_select_settings_container}async createImageLayerSettings(){const accentColor=this.darkMode?"maskEditor_accent_bg_dark":"maskEditor_accent_bg_light",image_layer_settings_container=this.createContainer(!0),image_layer_settings_title=this.createHeadline("Layers"),mask_layer_title=this.createContainerTitle("Mask Layer"),mask_layer_container=this.createContainer(!1);mask_layer_container.classList.add(accentColor),mask_layer_container.classList.add("maskEditor_layerRow");const mask_layer_visibility_checkbox=document.createElement("input");mask_layer_visibility_checkbox.setAttribute("type","checkbox"),mask_layer_visibility_checkbox.checked=!0,mask_layer_visibility_checkbox.classList.add("maskEditor_sidePanelLayerCheckbox"),mask_layer_visibility_checkbox.addEventListener("change",event=>{event.target.checked?this.maskCanvas.style.opacity=String(this.mask_opacity):this.maskCanvas.style.opacity="0"});var mask_layer_image_container=document.createElement("div");mask_layer_image_container.classList.add("maskEditor_sidePanelLayerPreviewContainer"),mask_layer_image_container.innerHTML='<svg viewBox="0 0 20 20" style="">   <path class="cls-1" d="M1.31,5.32v9.36c0,.55.45,1,1,1h15.38c.55,0,1-.45,1-1V5.32c0-.55-.45-1-1-1H2.31c-.55,0-1,.45-1,1ZM11.19,13.44c-2.91.94-5.57-1.72-4.63-4.63.34-1.05,1.19-1.9,2.24-2.24,2.91-.94,5.57,1.72,4.63,4.63-.34,1.05-1.19,1.9-2.24,2.24Z"/> </svg>';var blending_options=["black","white","negative"];const sidePanelDropdownAccent=this.darkMode?"maskEditor_sidePanelDropdown_dark":"maskEditor_sidePanelDropdown_light";var mask_layer_dropdown=document.createElement("select");mask_layer_dropdown.classList.add(sidePanelDropdownAccent),mask_layer_dropdown.classList.add(sidePanelDropdownAccent),blending_options.forEach(option=>{var option_element=document.createElement("option");option_element.value=option,option_element.innerText=option,mask_layer_dropdown.appendChild(option_element),option==this.maskBlendMode&&(option_element.selected=!0)}),mask_layer_dropdown.addEventListener("change",event=>{const selectedValue=event.target.value;this.maskBlendMode=selectedValue,this.updateMaskColor()}),mask_layer_container.appendChild(mask_layer_visibility_checkbox),mask_layer_container.appendChild(mask_layer_image_container),mask_layer_container.appendChild(mask_layer_dropdown);const mask_layer_opacity_sliderObj=this.createSlider("Mask Opacity",0,1,.01,this.mask_opacity,(event,value)=>{this.mask_opacity=parseFloat(value),this.maskCanvas.style.opacity=String(this.mask_opacity),this.mask_opacity==0?mask_layer_visibility_checkbox.checked=!1:mask_layer_visibility_checkbox.checked=!0});this.maskOpacitySlider=mask_layer_opacity_sliderObj.slider;const image_layer_title=this.createContainerTitle("Image Layer"),image_layer_container=this.createContainer(!1);image_layer_container.classList.add(accentColor),image_layer_container.classList.add("maskEditor_layerRow");const image_layer_visibility_checkbox=document.createElement("input");image_layer_visibility_checkbox.setAttribute("type","checkbox"),image_layer_visibility_checkbox.classList.add("maskEditor_sidePanelLayerCheckbox"),image_layer_visibility_checkbox.checked=!0,image_layer_visibility_checkbox.addEventListener("change",event=>{event.target.checked?this.imgCanvas.style.opacity="1":this.imgCanvas.style.opacity="0"});const image_layer_image_container=document.createElement("div");image_layer_image_container.classList.add("maskEditor_sidePanelLayerPreviewContainer");const image_layer_image=document.createElement("img");return image_layer_image.id="maskEditor_sidePanelImageLayerImage",image_layer_image.src=ComfyApp.clipspace?.imgs?.[ComfyApp.clipspace?.selectedIndex??0]?.src??"",this.sidebarImage=image_layer_image,image_layer_image_container.appendChild(image_layer_image),image_layer_container.appendChild(image_layer_visibility_checkbox),image_layer_container.appendChild(image_layer_image_container),image_layer_settings_container.appendChild(image_layer_settings_title),image_layer_settings_container.appendChild(mask_layer_title),image_layer_settings_container.appendChild(mask_layer_container),image_layer_settings_container.appendChild(mask_layer_opacity_sliderObj.container),image_layer_settings_container.appendChild(image_layer_title),image_layer_settings_container.appendChild(image_layer_container),image_layer_settings_container}createHeadline(title){var headline=document.createElement("h3");return headline.classList.add("maskEditor_sidePanelTitle"),headline.innerText=title,headline}createContainer(flexDirection){var container=document.createElement("div");return flexDirection?container.classList.add("maskEditor_sidePanelContainerColumn"):container.classList.add("maskEditor_sidePanelContainerRow"),container}createContainerTitle(title){var container_title=document.createElement("span");return container_title.classList.add("maskEditor_sidePanelSubTitle"),container_title.innerText=title,container_title}createSlider(title,min,max2,step,value,callback){var slider_container=this.createContainer(!0),slider_title=this.createContainerTitle(title),slider=document.createElement("input");return slider.classList.add("maskEditor_sidePanelBrushRange"),slider.setAttribute("type","range"),slider.setAttribute("min",String(min)),slider.setAttribute("max",String(max2)),slider.setAttribute("step",String(step)),slider.setAttribute("value",String(value)),slider.addEventListener("input",event=>{callback(event,event.target.value)}),slider_container.appendChild(slider_title),slider_container.appendChild(slider),{container:slider_container,slider}}createToggle(title,callback){var outer_Container=this.createContainer(!1),toggle_title=this.createContainerTitle(title),toggle_container=document.createElement("label");toggle_container.classList.add("maskEditor_sidePanelToggleContainer");var toggle_checkbox=document.createElement("input");toggle_checkbox.setAttribute("type","checkbox"),toggle_checkbox.classList.add("maskEditor_sidePanelToggleCheckbox"),toggle_checkbox.addEventListener("change",event=>{callback(event,event.target.checked)});var toggleAccentColor=this.darkMode?"maskEditor_toggle_bg_dark":"maskEditor_toggle_bg_light",toggle_switch=document.createElement("div");return toggle_switch.classList.add("maskEditor_sidePanelToggleSwitch"),toggle_switch.classList.add(toggleAccentColor),toggle_container.appendChild(toggle_checkbox),toggle_container.appendChild(toggle_switch),outer_Container.appendChild(toggle_title),outer_Container.appendChild(toggle_container),outer_Container}createDropdown(title,options,callback){const sidePanelDropdownAccent=this.darkMode?"maskEditor_sidePanelDropdown_dark":"maskEditor_sidePanelDropdown_light";var dropdown_container=this.createContainer(!1),dropdown_title=this.createContainerTitle(title),dropdown=document.createElement("select");return dropdown.classList.add(sidePanelDropdownAccent),dropdown.classList.add("maskEditor_containerDropdown"),options.forEach(option=>{var option_element=document.createElement("option");option_element.value=option,option_element.innerText=option,dropdown.appendChild(option_element)}),dropdown.addEventListener("change",event=>{callback(event,event.target.value)}),dropdown_container.appendChild(dropdown_title),dropdown_container.appendChild(dropdown),dropdown_container}createSeparator(){var separator=document.createElement("div");return separator.classList.add("maskEditor_sidePanelSeparator"),separator}async createTopBar(){const buttonAccentColor=this.darkMode?"maskEditor_topPanelButton_dark":"maskEditor_topPanelButton_light",iconButtonAccentColor=this.darkMode?"maskEditor_topPanelIconButton_dark":"maskEditor_topPanelIconButton_light";var top_bar=document.createElement("div");top_bar.id="maskEditor_topBar";var top_bar_title_container=document.createElement("div");top_bar_title_container.id="maskEditor_topBarTitleContainer";var top_bar_title=document.createElement("h1");top_bar_title.id="maskEditor_topBarTitle",top_bar_title.innerText="ComfyUI",top_bar_title_container.appendChild(top_bar_title);var top_bar_shortcuts_container=document.createElement("div");top_bar_shortcuts_container.id="maskEditor_topBarShortcutsContainer";var top_bar_undo_button=document.createElement("div");top_bar_undo_button.id="maskEditor_topBarUndoButton",top_bar_undo_button.classList.add(iconButtonAccentColor),top_bar_undo_button.innerHTML='<svg viewBox="0 0 15 15"><path d="M8.77,12.18c-.25,0-.46-.2-.46-.46s.2-.46.46-.46c1.47,0,2.67-1.2,2.67-2.67,0-1.57-1.34-2.67-3.26-2.67h-3.98l1.43,1.43c.18.18.18.47,0,.64-.18.18-.47.18-.64,0l-2.21-2.21c-.18-.18-.18-.47,0-.64l2.21-2.21c.18-.18.47-.18.64,0,.18.18.18.47,0,.64l-1.43,1.43h3.98c2.45,0,4.17,1.47,4.17,3.58,0,1.97-1.61,3.58-3.58,3.58Z"></path> </svg>',top_bar_undo_button.addEventListener("click",()=>{this.messageBroker.publish("undo")});var top_bar_redo_button=document.createElement("div");top_bar_redo_button.id="maskEditor_topBarRedoButton",top_bar_redo_button.classList.add(iconButtonAccentColor),top_bar_redo_button.innerHTML='<svg viewBox="0 0 15 15"> <path class="cls-1" d="M6.23,12.18c-1.97,0-3.58-1.61-3.58-3.58,0-2.11,1.71-3.58,4.17-3.58h3.98l-1.43-1.43c-.18-.18-.18-.47,0-.64.18-.18.46-.18.64,0l2.21,2.21c.09.09.13.2.13.32s-.05.24-.13.32l-2.21,2.21c-.18.18-.47.18-.64,0-.18-.18-.18-.47,0-.64l1.43-1.43h-3.98c-1.92,0-3.26,1.1-3.26,2.67,0,1.47,1.2,2.67,2.67,2.67.25,0,.46.2.46.46s-.2.46-.46.46Z"/></svg>',top_bar_redo_button.addEventListener("click",()=>{this.messageBroker.publish("redo")});var top_bar_invert_button=document.createElement("button");top_bar_invert_button.id="maskEditor_topBarInvertButton",top_bar_invert_button.classList.add(buttonAccentColor),top_bar_invert_button.innerText="Invert",top_bar_invert_button.addEventListener("click",()=>{this.messageBroker.publish("invert")});var top_bar_clear_button=document.createElement("button");top_bar_clear_button.id="maskEditor_topBarClearButton",top_bar_clear_button.classList.add(buttonAccentColor),top_bar_clear_button.innerText="Clear",top_bar_clear_button.addEventListener("click",()=>{this.maskCtx.clearRect(0,0,this.maskCanvas.width,this.maskCanvas.height),this.messageBroker.publish("saveState")});var top_bar_save_button=document.createElement("button");top_bar_save_button.id="maskEditor_topBarSaveButton",top_bar_save_button.classList.add(buttonAccentColor),top_bar_save_button.innerText="Save",this.saveButton=top_bar_save_button,top_bar_save_button.addEventListener("click",()=>{this.maskEditor.save()});var top_bar_cancel_button=document.createElement("button");return top_bar_cancel_button.id="maskEditor_topBarCancelButton",top_bar_cancel_button.classList.add(buttonAccentColor),top_bar_cancel_button.innerText="Cancel",top_bar_cancel_button.addEventListener("click",()=>{this.maskEditor.close()}),top_bar_shortcuts_container.appendChild(top_bar_undo_button),top_bar_shortcuts_container.appendChild(top_bar_redo_button),top_bar_shortcuts_container.appendChild(top_bar_invert_button),top_bar_shortcuts_container.appendChild(top_bar_clear_button),top_bar_shortcuts_container.appendChild(top_bar_save_button),top_bar_shortcuts_container.appendChild(top_bar_cancel_button),top_bar.appendChild(top_bar_title_container),top_bar.appendChild(top_bar_shortcuts_container),top_bar}createToolPanel(){var tool_panel=document.createElement("div");tool_panel.id="maskEditor_toolPanel",this.toolPanel=tool_panel;var toolPanelHoverAccent=this.darkMode?"maskEditor_toolPanelContainerDark":"maskEditor_toolPanelContainerLight",toolElements=[],toolPanel_brushToolContainer=document.createElement("div");toolPanel_brushToolContainer.classList.add("maskEditor_toolPanelContainer"),toolPanel_brushToolContainer.classList.add("maskEditor_toolPanelContainerSelected"),toolPanel_brushToolContainer.classList.add(toolPanelHoverAccent),toolPanel_brushToolContainer.innerHTML=`
    <svg viewBox="0 0 44 44">
      <path class="cls-1" d="M34,13.93c0,.47-.19.94-.55,1.31l-13.02,13.04c-.09.07-.18.15-.27.22-.07-1.39-1.21-2.48-2.61-2.49.07-.12.16-.24.27-.34l13.04-13.04c.72-.72,1.89-.72,2.6,0,.35.35.55.83.55,1.3Z"/>
      <path class="cls-1" d="M19.64,29.03c0,4.46-6.46,3.18-9.64,0,3.3-.47,4.75-2.58,7.06-2.58,1.43,0,2.58,1.16,2.58,2.58Z"/>
    </svg>
    `,toolElements.push(toolPanel_brushToolContainer),toolPanel_brushToolContainer.addEventListener("click",()=>{this.messageBroker.publish("setTool","pen");for(let toolElement of toolElements)toolElement!=toolPanel_brushToolContainer?toolElement.classList.remove("maskEditor_toolPanelContainerSelected"):(toolElement.classList.add("maskEditor_toolPanelContainerSelected"),this.brushSettingsHTML.style.display="flex",this.colorSelectSettingsHTML.style.display="none",this.paintBucketSettingsHTML.style.display="none");this.messageBroker.publish("setTool","pen"),this.pointerZone.style.cursor="none"});var toolPanel_brushToolIndicator=document.createElement("div");toolPanel_brushToolIndicator.classList.add("maskEditor_toolPanelIndicator"),toolPanel_brushToolContainer.appendChild(toolPanel_brushToolIndicator);var toolPanel_eraserToolContainer=document.createElement("div");toolPanel_eraserToolContainer.classList.add("maskEditor_toolPanelContainer"),toolPanel_eraserToolContainer.classList.add(toolPanelHoverAccent),toolPanel_eraserToolContainer.innerHTML=`
      <svg viewBox="0 0 44 44">
        <g>
          <rect class="cls-2" x="16.68" y="10" width="10.63" height="24" rx="1.16" ry="1.16" transform="translate(22 -9.11) rotate(45)"/>
          <path class="cls-1" d="M17.27,34.27c-.42,0-.85-.16-1.17-.48l-5.88-5.88c-.31-.31-.48-.73-.48-1.17s.17-.86.48-1.17l15.34-15.34c.62-.62,1.72-.62,2.34,0l5.88,5.88c.65.65.65,1.7,0,2.34l-15.34,15.34c-.32.32-.75.48-1.17.48ZM26.73,10.73c-.18,0-.34.07-.46.19l-15.34,15.34c-.12.12-.19.29-.19.46s.07.34.19.46l5.88,5.88c.26.26.67.26.93,0l15.34-15.34c.26-.26.26-.67,0-.93l-5.88-5.88c-.12-.12-.29-.19-.46-.19Z"/>
        </g>
        <path class="cls-3" d="M20.33,11.03h8.32c.64,0,1.16.52,1.16,1.16v15.79h-10.63v-15.79c0-.64.52-1.16,1.16-1.16Z" transform="translate(20.97 -11.61) rotate(45)"/>
      </svg>
    `,toolElements.push(toolPanel_eraserToolContainer),toolPanel_eraserToolContainer.addEventListener("click",()=>{this.messageBroker.publish("setTool","eraser");for(let toolElement of toolElements)toolElement!=toolPanel_eraserToolContainer?toolElement.classList.remove("maskEditor_toolPanelContainerSelected"):(toolElement.classList.add("maskEditor_toolPanelContainerSelected"),this.brushSettingsHTML.style.display="flex",this.colorSelectSettingsHTML.style.display="none",this.paintBucketSettingsHTML.style.display="none");this.messageBroker.publish("setTool","eraser"),this.pointerZone.style.cursor="none"});var toolPanel_eraserToolIndicator=document.createElement("div");toolPanel_eraserToolIndicator.classList.add("maskEditor_toolPanelIndicator"),toolPanel_eraserToolContainer.appendChild(toolPanel_eraserToolIndicator);var toolPanel_paintBucketToolContainer=document.createElement("div");toolPanel_paintBucketToolContainer.classList.add("maskEditor_toolPanelContainer"),toolPanel_paintBucketToolContainer.classList.add(toolPanelHoverAccent),toolPanel_paintBucketToolContainer.innerHTML=`
    <svg viewBox="0 0 44 44">
      <path class="cls-1" d="M33.4,21.76l-11.42,11.41-.04.05c-.61.61-1.6.61-2.21,0l-8.91-8.91c-.61-.61-.61-1.6,0-2.21l.04-.05.3-.29h22.24Z"/>
      <path class="cls-1" d="M20.83,34.17c-.55,0-1.07-.21-1.46-.6l-8.91-8.91c-.8-.8-.8-2.11,0-2.92l11.31-11.31c.8-.8,2.11-.8,2.92,0l8.91,8.91c.39.39.6.91.6,1.46s-.21,1.07-.6,1.46l-11.31,11.31c-.39.39-.91.6-1.46.6ZM23.24,10.83c-.27,0-.54.1-.75.31l-11.31,11.31c-.41.41-.41,1.09,0,1.5l8.91,8.91c.4.4,1.1.4,1.5,0l11.31-11.31c.2-.2.31-.47.31-.75s-.11-.55-.31-.75l-8.91-8.91c-.21-.21-.48-.31-.75-.31Z"/>
      <path class="cls-1" d="M34.28,26.85c0,.84-.68,1.52-1.52,1.52s-1.52-.68-1.52-1.52,1.52-2.86,1.52-2.86c0,0,1.52,2.02,1.52,2.86Z"/>
    </svg>
    `,toolElements.push(toolPanel_paintBucketToolContainer),toolPanel_paintBucketToolContainer.addEventListener("click",()=>{this.messageBroker.publish("setTool","paintBucket");for(let toolElement of toolElements)toolElement!=toolPanel_paintBucketToolContainer?toolElement.classList.remove("maskEditor_toolPanelContainerSelected"):(toolElement.classList.add("maskEditor_toolPanelContainerSelected"),this.brushSettingsHTML.style.display="none",this.colorSelectSettingsHTML.style.display="none",this.paintBucketSettingsHTML.style.display="flex");this.messageBroker.publish("setTool","paintBucket"),this.pointerZone.style.cursor="url('/cursor/paintBucket.png') 30 25, auto",this.brush.style.opacity="0"});var toolPanel_paintBucketToolIndicator=document.createElement("div");toolPanel_paintBucketToolIndicator.classList.add("maskEditor_toolPanelIndicator"),toolPanel_paintBucketToolContainer.appendChild(toolPanel_paintBucketToolIndicator);var toolPanel_colorSelectToolContainer=document.createElement("div");toolPanel_colorSelectToolContainer.classList.add("maskEditor_toolPanelContainer"),toolPanel_colorSelectToolContainer.classList.add(toolPanelHoverAccent),toolPanel_colorSelectToolContainer.innerHTML=`
    <svg viewBox="0 0 44 44">
      <path class="cls-1" d="M30.29,13.72c-1.09-1.1-2.85-1.09-3.94,0l-2.88,2.88-.75-.75c-.2-.19-.51-.19-.71,0-.19.2-.19.51,0,.71l1.4,1.4-9.59,9.59c-.35.36-.54.82-.54,1.32,0,.14,0,.28.05.41-.05.04-.1.08-.15.13-.39.39-.39,1.01,0,1.4.38.39,1.01.39,1.4,0,.04-.04.08-.09.11-.13.14.04.3.06.45.06.5,0,.97-.19,1.32-.55l9.59-9.59,1.38,1.38c.1.09.22.14.35.14s.26-.05.35-.14c.2-.2.2-.52,0-.71l-.71-.72,2.88-2.89c1.08-1.08,1.08-2.85-.01-3.94ZM19.43,25.82h-2.46l7.15-7.15,1.23,1.23-5.92,5.92Z"/>
    </svg>
    `,toolElements.push(toolPanel_colorSelectToolContainer),toolPanel_colorSelectToolContainer.addEventListener("click",()=>{this.messageBroker.publish("setTool","colorSelect");for(let toolElement of toolElements)toolElement!=toolPanel_colorSelectToolContainer?toolElement.classList.remove("maskEditor_toolPanelContainerSelected"):(toolElement.classList.add("maskEditor_toolPanelContainerSelected"),this.brushSettingsHTML.style.display="none",this.paintBucketSettingsHTML.style.display="none",this.colorSelectSettingsHTML.style.display="flex");this.messageBroker.publish("setTool","colorSelect"),this.pointerZone.style.cursor="url('/cursor/colorSelect.png') 15 25, auto",this.brush.style.opacity="0"});var toolPanel_colorSelectToolIndicator=document.createElement("div");toolPanel_colorSelectToolIndicator.classList.add("maskEditor_toolPanelIndicator"),toolPanel_colorSelectToolContainer.appendChild(toolPanel_colorSelectToolIndicator);var toolPanel_zoomIndicator=document.createElement("div");toolPanel_zoomIndicator.classList.add("maskEditor_toolPanelZoomIndicator"),toolPanel_zoomIndicator.classList.add(toolPanelHoverAccent);var toolPanel_zoomText=document.createElement("span");toolPanel_zoomText.id="maskEditor_toolPanelZoomText",toolPanel_zoomText.innerText="100%",this.zoomTextHTML=toolPanel_zoomText;var toolPanel_DimensionsText=document.createElement("span");return toolPanel_DimensionsText.id="maskEditor_toolPanelDimensionsText",toolPanel_DimensionsText.innerText=" ",this.dimensionsTextHTML=toolPanel_DimensionsText,toolPanel_zoomIndicator.appendChild(toolPanel_zoomText),toolPanel_zoomIndicator.appendChild(toolPanel_DimensionsText),toolPanel_zoomIndicator.addEventListener("click",()=>{this.messageBroker.publish("resetZoom")}),tool_panel.appendChild(toolPanel_brushToolContainer),tool_panel.appendChild(toolPanel_eraserToolContainer),tool_panel.appendChild(toolPanel_paintBucketToolContainer),tool_panel.appendChild(toolPanel_colorSelectToolContainer),tool_panel.appendChild(toolPanel_zoomIndicator),tool_panel}createPointerZone(){const pointer_zone=document.createElement("div");return pointer_zone.id="maskEditor_pointerZone",this.pointerZone=pointer_zone,pointer_zone.addEventListener("pointerdown",event=>{this.messageBroker.publish("pointerDown",event)}),pointer_zone.addEventListener("pointermove",event=>{this.messageBroker.publish("pointerMove",event)}),pointer_zone.addEventListener("pointerup",event=>{this.messageBroker.publish("pointerUp",event)}),pointer_zone.addEventListener("pointerleave",event=>{this.brush.style.opacity="0",this.pointerZone.style.cursor=""}),pointer_zone.addEventListener("touchstart",event=>{this.messageBroker.publish("handleTouchStart",event)}),pointer_zone.addEventListener("touchmove",event=>{this.messageBroker.publish("handleTouchMove",event)}),pointer_zone.addEventListener("touchend",event=>{this.messageBroker.publish("handleTouchEnd",event)}),pointer_zone.addEventListener("wheel",event=>this.messageBroker.publish("wheel",event)),pointer_zone.addEventListener("pointerenter",async event=>{this.updateCursor()}),pointer_zone}async screenToCanvas(clientPoint){const zoomRatio=await this.messageBroker.pull("zoomRatio"),canvasRect=this.maskCanvas.getBoundingClientRect(),offsetX=clientPoint.x-canvasRect.left+this.toolPanel.clientWidth,offsetY=clientPoint.y-canvasRect.top+44,x=offsetX/zoomRatio,y=offsetY/zoomRatio;return{x,y}}setEventHandler(){this.maskCanvas.addEventListener("contextmenu",event=>{event.preventDefault()}),this.rootElement.addEventListener("contextmenu",event=>{event.preventDefault()}),this.rootElement.addEventListener("dragstart",event=>{event.ctrlKey&&event.preventDefault()})}async createBrush(){var brush=document.createElement("div");const brushSettings=await this.messageBroker.pull("brushSettings");brush.id="maskEditor_brush";var brush_preview_gradient=document.createElement("div");return brush_preview_gradient.id="maskEditor_brushPreviewGradient",brush.appendChild(brush_preview_gradient),this.brush=brush,this.brushPreviewGradient=brush_preview_gradient,brush}async setImages(imgCanvas){const imgCtx=imgCanvas.getContext("2d",{willReadFrequently:!0}),maskCtx=this.maskCtx,maskCanvas=this.maskCanvas;imgCtx.clearRect(0,0,this.imgCanvas.width,this.imgCanvas.height),maskCtx.clearRect(0,0,this.maskCanvas.width,this.maskCanvas.height);const alpha_url=new URL(ComfyApp.clipspace?.imgs?.[ComfyApp.clipspace?.selectedIndex??0]?.src??"");alpha_url.searchParams.delete("channel"),alpha_url.searchParams.delete("preview"),alpha_url.searchParams.set("channel","a");let mask_image=await this.loadImage(alpha_url);if(!ComfyApp.clipspace?.imgs?.[ComfyApp.clipspace?.selectedIndex??0]?.src)throw new Error("Unable to access image source - clipspace or image is null");const rgb_url=new URL(ComfyApp.clipspace.imgs[ComfyApp.clipspace.selectedIndex].src);this.imageURL=rgb_url,console.log(rgb_url),rgb_url.searchParams.delete("channel"),rgb_url.searchParams.set("channel","rgb"),this.image=new Image,this.image=await new Promise((resolve,reject)=>{const img=new Image;img.onload=()=>resolve(img),img.onerror=reject,img.src=rgb_url.toString()}),maskCanvas.width=this.image.width,maskCanvas.height=this.image.height,this.dimensionsTextHTML.innerText=`${this.image.width}x${this.image.height}`,await this.invalidateCanvas(this.image,mask_image),this.messageBroker.publish("initZoomPan",[this.image,this.rootElement])}async invalidateCanvas(orig_image,mask_image){this.imgCanvas.width=orig_image.width,this.imgCanvas.height=orig_image.height,this.maskCanvas.width=orig_image.width,this.maskCanvas.height=orig_image.height;let imgCtx=this.imgCanvas.getContext("2d",{willReadFrequently:!0}),maskCtx=this.maskCanvas.getContext("2d",{willReadFrequently:!0});imgCtx.drawImage(orig_image,0,0,orig_image.width,orig_image.height),await this.prepare_mask(mask_image,this.maskCanvas,maskCtx,await this.getMaskColor())}async prepare_mask(image,maskCanvas,maskCtx,maskColor){maskCtx.drawImage(image,0,0,maskCanvas.width,maskCanvas.height);const maskData=maskCtx.getImageData(0,0,maskCanvas.width,maskCanvas.height);for(let i=0;i<maskData.data.length;i+=4){const alpha=maskData.data[i+3];maskData.data[i]=maskColor.r,maskData.data[i+1]=maskColor.g,maskData.data[i+2]=maskColor.b,maskData.data[i+3]=255-alpha}maskCtx.globalCompositeOperation="source-over",maskCtx.putImageData(maskData,0,0)}async updateMaskColor(){const maskCanvasStyle=this.getMaskCanvasStyle();this.maskCanvas.style.mixBlendMode=maskCanvasStyle.mixBlendMode,this.maskCanvas.style.opacity=maskCanvasStyle.opacity.toString();const maskColor=await this.getMaskColor();this.maskCtx.fillStyle=`rgb(${maskColor.r}, ${maskColor.g}, ${maskColor.b})`,this.setCanvasBackground();const maskData=this.maskCtx.getImageData(0,0,this.maskCanvas.width,this.maskCanvas.height);for(let i=0;i<maskData.data.length;i+=4)maskData.data[i]=maskColor.r,maskData.data[i+1]=maskColor.g,maskData.data[i+2]=maskColor.b;this.maskCtx.putImageData(maskData,0,0)}getMaskCanvasStyle(){return this.maskBlendMode==="negative"?{mixBlendMode:"difference",opacity:"1"}:{mixBlendMode:"initial",opacity:this.mask_opacity}}detectLightMode(){this.darkMode=document.body.classList.contains("dark-theme")}loadImage(imagePath){return new Promise((resolve,reject)=>{const image=new Image;image.onload=function(){resolve(image)},image.onerror=function(error){reject(error)},image.src=imagePath.href})}async updateBrushPreview(){const cursorPoint=await this.messageBroker.pull("cursorPoint"),pan_offset=await this.messageBroker.pull("panOffset"),brushSettings=await this.messageBroker.pull("brushSettings"),zoom_ratio=await this.messageBroker.pull("zoomRatio"),centerX=cursorPoint.x+pan_offset.x,centerY=cursorPoint.y+pan_offset.y,brush=this.brush,hardness=brushSettings.hardness,extendedSize=brushSettings.size*(2-hardness)*2*zoom_ratio;if(this.brushSizeSlider.value=String(brushSettings.size),this.brushHardnessSlider.value=String(hardness),brush.style.width=extendedSize+"px",brush.style.height=extendedSize+"px",brush.style.left=centerX-extendedSize/2+"px",brush.style.top=centerY-extendedSize/2+"px",hardness===1){this.brushPreviewGradient.style.background="rgba(255, 0, 0, 0.5)";return}const opacityStop=hardness/4+.25;this.brushPreviewGradient.style.background=`
        radial-gradient(
            circle,
            rgba(255, 0, 0, 0.5) 0%,
            rgba(255, 0, 0, ${opacityStop}) ${hardness*100}%,
            rgba(255, 0, 0, 0) 100%
        )
    `}getMaskBlendMode(){return this.maskBlendMode}setSidebarImage(){this.sidebarImage.src=this.imageURL.href}async getMaskColor(){return this.maskBlendMode==="black"?{r:0,g:0,b:0}:this.maskBlendMode==="white"?{r:255,g:255,b:255}:this.maskBlendMode==="negative"?{r:255,g:255,b:255}:{r:0,g:0,b:0}}async getMaskFillStyle(){const maskColor=await this.getMaskColor();return"rgb("+maskColor.r+","+maskColor.g+","+maskColor.b+")"}async setCanvasBackground(){this.maskBlendMode==="white"?this.canvasBackground.style.background="black":this.canvasBackground.style.background="white"}getMaskCanvas(){return this.maskCanvas}getImgCanvas(){return this.imgCanvas}getImage(){return this.image}setBrushOpacity(opacity){this.brush.style.opacity=String(opacity)}setSaveButtonEnabled(enabled){this.saveButton.disabled=!enabled}setSaveButtonText(text){this.saveButton.innerText=text}handlePaintBucketCursor(isPaintBucket){isPaintBucket?this.pointerZone.style.cursor="url('/cursor/paintBucket.png') 30 25, auto":this.pointerZone.style.cursor="none"}handlePanCursor(isPanning){isPanning?this.pointerZone.style.cursor="grabbing":this.pointerZone.style.cursor="none"}setBrushVisibility(visible){this.brush.style.opacity=visible?"1":"0"}setBrushPreviewGradientVisibility(visible){this.brushPreviewGradient.style.display=visible?"block":"none"}async updateCursor(){const currentTool=await this.messageBroker.pull("currentTool");currentTool==="paintBucket"?(this.pointerZone.style.cursor="url('/cursor/paintBucket.png') 30 25, auto",this.setBrushOpacity(0)):currentTool==="colorSelect"?(this.pointerZone.style.cursor="url('/cursor/colorSelect.png') 15 25, auto",this.setBrushOpacity(0)):(this.pointerZone.style.cursor="none",this.setBrushOpacity(1)),this.updateBrushPreview(),this.setBrushPreviewGradientVisibility(!1)}setZoomText(zoomText){this.zoomTextHTML.innerText=zoomText}setDimensionsText(dimensionsText){this.dimensionsTextHTML.innerText=dimensionsText}}class ToolManager{static{__name(this,"ToolManager")}maskEditor;messageBroker;mouseDownPoint=null;currentTool="pen";isAdjustingBrush=!1;constructor(maskEditor){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.addListeners(),this.addPullTopics()}addListeners(){this.messageBroker.subscribe("setTool",async tool=>{this.setTool(tool)}),this.messageBroker.subscribe("pointerDown",async event=>{this.handlePointerDown(event)}),this.messageBroker.subscribe("pointerMove",async event=>{this.handlePointerMove(event)}),this.messageBroker.subscribe("pointerUp",async event=>{this.handlePointerUp(event)}),this.messageBroker.subscribe("wheel",async event=>{this.handleWheelEvent(event)})}async addPullTopics(){this.messageBroker.createPullTopic("currentTool",async()=>this.getCurrentTool())}setTool(tool){this.currentTool=tool,tool!="colorSelect"&&this.messageBroker.publish("clearLastPoint")}getCurrentTool(){return this.currentTool}async handlePointerDown(event){if(event.preventDefault(),event.pointerType!="touch"){var isSpacePressed=await this.messageBroker.pull("isKeyPressed"," ");if(event.buttons===4||event.buttons===1&&isSpacePressed){this.messageBroker.publish("panStart",event),this.messageBroker.publish("setBrushVisibility",!1);return}if(this.currentTool==="paintBucket"&&event.button===0){const offset={x:event.offsetX,y:event.offsetY},coords_canvas=await this.messageBroker.pull("screenToCanvas",offset);this.messageBroker.publish("paintBucketFill",coords_canvas),this.messageBroker.publish("saveState");return}if(this.currentTool==="colorSelect"&&event.button===0){const offset={x:event.offsetX,y:event.offsetY},coords_canvas=await this.messageBroker.pull("screenToCanvas",offset);this.messageBroker.publish("colorSelectFill",coords_canvas);return}if(event.altKey&&event.button===2){this.isAdjustingBrush=!0,this.messageBroker.publish("brushAdjustmentStart",event);return}var isDrawingTool=["pen","eraser"].includes(this.currentTool);if([0,2].includes(event.button)&&isDrawingTool){this.messageBroker.publish("drawStart",event);return}}}async handlePointerMove(event){if(event.preventDefault(),event.pointerType=="touch")return;const newCursorPoint={x:event.clientX,y:event.clientY};this.messageBroker.publish("cursorPoint",newCursorPoint);var isSpacePressed=await this.messageBroker.pull("isKeyPressed"," ");if(this.messageBroker.publish("updateBrushPreview"),event.buttons===4||event.buttons===1&&isSpacePressed){this.messageBroker.publish("panMove",event);return}var isDrawingTool=["pen","eraser"].includes(this.currentTool);if(isDrawingTool){if(this.isAdjustingBrush&&(this.currentTool==="pen"||this.currentTool==="eraser")&&event.altKey&&event.buttons===2){this.messageBroker.publish("brushAdjustment",event);return}if(event.buttons==1||event.buttons==2){this.messageBroker.publish("draw",event);return}}}handlePointerUp(event){this.messageBroker.publish("panCursor",!1),event.pointerType!=="touch"&&(this.messageBroker.publish("updateCursor"),this.isAdjustingBrush=!1,this.messageBroker.publish("drawEnd",event),this.mouseDownPoint=null)}handleWheelEvent(event){this.messageBroker.publish("zoom",event);const newCursorPoint={x:event.clientX,y:event.clientY};this.messageBroker.publish("cursorPoint",newCursorPoint)}}class PanAndZoomManager{static{__name(this,"PanAndZoomManager")}maskEditor;messageBroker;DOUBLE_TAP_DELAY=300;lastTwoFingerTap=0;isTouchZooming=!1;lastTouchZoomDistance=0;lastTouchMidPoint={x:0,y:0};lastTouchPoint={x:0,y:0};zoom_ratio=1;interpolatedZoomRatio=1;pan_offset={x:0,y:0};mouseDownPoint=null;initialPan={x:0,y:0};canvasContainer=null;maskCanvas=null;rootElement=null;image=null;imageRootWidth=0;imageRootHeight=0;cursorPoint={x:0,y:0};constructor(maskEditor){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.addListeners(),this.addPullTopics()}addListeners(){this.messageBroker.subscribe("initZoomPan",async args=>{await this.initializeCanvasPanZoom(args[0],args[1])}),this.messageBroker.subscribe("panStart",async event=>{this.handlePanStart(event)}),this.messageBroker.subscribe("panMove",async event=>{this.handlePanMove(event)}),this.messageBroker.subscribe("zoom",async event=>{this.zoom(event)}),this.messageBroker.subscribe("cursorPoint",async point=>{this.updateCursorPosition(point)}),this.messageBroker.subscribe("handleTouchStart",async event=>{this.handleTouchStart(event)}),this.messageBroker.subscribe("handleTouchMove",async event=>{this.handleTouchMove(event)}),this.messageBroker.subscribe("handleTouchEnd",async event=>{this.handleTouchEnd(event)}),this.messageBroker.subscribe("resetZoom",async()=>{this.interpolatedZoomRatio!==1&&await this.smoothResetView()})}addPullTopics(){this.messageBroker.createPullTopic("cursorPoint",async()=>this.cursorPoint),this.messageBroker.createPullTopic("zoomRatio",async()=>this.zoom_ratio),this.messageBroker.createPullTopic("panOffset",async()=>this.pan_offset)}handleTouchStart(event){if(event.preventDefault(),event.touches[0].touchType!=="stylus")if(this.messageBroker.publish("setBrushVisibility",!1),event.touches.length===2){const currentTime=new Date().getTime();if(currentTime-this.lastTwoFingerTap<this.DOUBLE_TAP_DELAY)this.handleDoubleTap(),this.lastTwoFingerTap=0;else{this.lastTwoFingerTap=currentTime,this.isTouchZooming=!0,this.lastTouchZoomDistance=this.getTouchDistance(event.touches);const midpoint=this.getTouchMidpoint(event.touches);this.lastTouchMidPoint=midpoint}}else event.touches.length===1&&(this.lastTouchPoint={x:event.touches[0].clientX,y:event.touches[0].clientY})}async handleTouchMove(event){if(event.preventDefault(),event.touches[0].touchType!=="stylus")if(this.lastTwoFingerTap=0,this.isTouchZooming&&event.touches.length===2){const newDistance=this.getTouchDistance(event.touches),zoomFactor=newDistance/this.lastTouchZoomDistance,oldZoom=this.zoom_ratio;this.zoom_ratio=Math.max(.2,Math.min(10,this.zoom_ratio*zoomFactor));const newZoom=this.zoom_ratio,midpoint=this.getTouchMidpoint(event.touches);if(this.lastTouchMidPoint){const deltaX=midpoint.x-this.lastTouchMidPoint.x,deltaY=midpoint.y-this.lastTouchMidPoint.y;this.pan_offset.x+=deltaX,this.pan_offset.y+=deltaY}this.maskCanvas===null&&(this.maskCanvas=await this.messageBroker.pull("maskCanvas"));const rect=this.maskCanvas.getBoundingClientRect(),touchX=midpoint.x-rect.left,touchY=midpoint.y-rect.top,scaleFactor=newZoom/oldZoom;this.pan_offset.x+=touchX-touchX*scaleFactor,this.pan_offset.y+=touchY-touchY*scaleFactor,this.invalidatePanZoom(),this.lastTouchZoomDistance=newDistance,this.lastTouchMidPoint=midpoint}else event.touches.length===1&&this.handleSingleTouchPan(event.touches[0])}handleTouchEnd(event){event.preventDefault(),!(event.touches.length===0&&event.touches[0].touchType==="stylus")&&(this.isTouchZooming=!1,this.lastTouchMidPoint={x:0,y:0},event.touches.length===0?this.lastTouchPoint={x:0,y:0}:event.touches.length===1&&(this.lastTouchPoint={x:event.touches[0].clientX,y:event.touches[0].clientY}))}getTouchDistance(touches){const dx=touches[0].clientX-touches[1].clientX,dy=touches[0].clientY-touches[1].clientY;return Math.sqrt(dx*dx+dy*dy)}getTouchMidpoint(touches){return{x:(touches[0].clientX+touches[1].clientX)/2,y:(touches[0].clientY+touches[1].clientY)/2}}async handleSingleTouchPan(touch){if(this.lastTouchPoint===null){this.lastTouchPoint={x:touch.clientX,y:touch.clientY};return}const deltaX=touch.clientX-this.lastTouchPoint.x,deltaY=touch.clientY-this.lastTouchPoint.y;this.pan_offset.x+=deltaX,this.pan_offset.y+=deltaY,await this.invalidatePanZoom(),this.lastTouchPoint={x:touch.clientX,y:touch.clientY}}updateCursorPosition(clientPoint){var cursorX=clientPoint.x-this.pan_offset.x,cursorY=clientPoint.y-this.pan_offset.y;this.cursorPoint={x:cursorX,y:cursorY}}handleDoubleTap(){this.messageBroker.publish("undo")}async zoom(event){const cursorPoint={x:event.clientX,y:event.clientY},oldZoom=this.zoom_ratio,zoomFactor=event.deltaY<0?1.1:.9;this.zoom_ratio=Math.max(.2,Math.min(10,this.zoom_ratio*zoomFactor));const newZoom=this.zoom_ratio,maskCanvas=await this.messageBroker.pull("maskCanvas"),rect=maskCanvas.getBoundingClientRect(),mouseX=cursorPoint.x-rect.left,mouseY=cursorPoint.y-rect.top;console.log(oldZoom,newZoom);const scaleFactor=newZoom/oldZoom;this.pan_offset.x+=mouseX-mouseX*scaleFactor,this.pan_offset.y+=mouseY-mouseY*scaleFactor,await this.invalidatePanZoom();const zoomRatio=maskCanvas.clientWidth/this.imageRootWidth;this.interpolatedZoomRatio=zoomRatio,this.messageBroker.publish("setZoomText",`${Math.round(zoomRatio*100)}%`),this.updateCursorPosition(cursorPoint),requestAnimationFrame(()=>{this.messageBroker.publish("updateBrushPreview")})}async smoothResetView(duration=500){const startZoom=this.zoom_ratio,startPan={...this.pan_offset},sidePanelWidth=220,toolPanelWidth=64,topBarHeight=44,availableWidth=this.rootElement.clientWidth-sidePanelWidth-toolPanelWidth,availableHeight=this.rootElement.clientHeight-topBarHeight,zoomRatioWidth=availableWidth/this.image.width,zoomRatioHeight=availableHeight/this.image.height,targetZoom=Math.min(zoomRatioWidth,zoomRatioHeight),aspectRatio=this.image.width/this.image.height;let finalWidth=0,finalHeight=0;const targetPan={x:toolPanelWidth,y:topBarHeight};zoomRatioHeight>zoomRatioWidth?(finalWidth=availableWidth,finalHeight=finalWidth/aspectRatio,targetPan.y=(availableHeight-finalHeight)/2+topBarHeight):(finalHeight=availableHeight,finalWidth=finalHeight*aspectRatio,targetPan.x=(availableWidth-finalWidth)/2+toolPanelWidth);const startTime=performance.now(),animate=__name(currentTime=>{const elapsed=currentTime-startTime,progress=Math.min(elapsed/duration,1),eased=1-Math.pow(1-progress,3),currentZoom=startZoom+(targetZoom-startZoom)*eased;this.zoom_ratio=currentZoom,this.pan_offset.x=startPan.x+(targetPan.x-startPan.x)*eased,this.pan_offset.y=startPan.y+(targetPan.y-startPan.y)*eased,this.invalidatePanZoom();const interpolatedZoomRatio=startZoom+(1-startZoom)*eased;this.messageBroker.publish("setZoomText",`${Math.round(interpolatedZoomRatio*100)}%`),progress<1&&requestAnimationFrame(animate)},"animate");requestAnimationFrame(animate),this.interpolatedZoomRatio=1}async initializeCanvasPanZoom(image,rootElement){let sidePanelWidth=220;const toolPanelWidth=64;let topBarHeight=44;this.rootElement=rootElement;let availableWidth=rootElement.clientWidth-sidePanelWidth-toolPanelWidth,availableHeight=rootElement.clientHeight-topBarHeight,zoomRatioWidth=availableWidth/image.width,zoomRatioHeight=availableHeight/image.height,aspectRatio=image.width/image.height,finalWidth=0,finalHeight=0,pan_offset={x:toolPanelWidth,y:topBarHeight};zoomRatioHeight>zoomRatioWidth?(finalWidth=availableWidth,finalHeight=finalWidth/aspectRatio,pan_offset.y=(availableHeight-finalHeight)/2+topBarHeight):(finalHeight=availableHeight,finalWidth=finalHeight*aspectRatio,pan_offset.x=(availableWidth-finalWidth)/2+toolPanelWidth),this.image===null&&(this.image=image),this.imageRootWidth=finalWidth,this.imageRootHeight=finalHeight,this.zoom_ratio=Math.min(zoomRatioWidth,zoomRatioHeight),this.pan_offset=pan_offset,await this.invalidatePanZoom()}async invalidatePanZoom(){if(!this.image?.width||!this.image?.height||!this.pan_offset||!this.zoom_ratio){console.warn("Missing required properties for pan/zoom");return}const raw_width=this.image.width*this.zoom_ratio,raw_height=this.image.height*this.zoom_ratio;this.canvasContainer??=await this.messageBroker?.pull("getCanvasContainer"),this.canvasContainer&&Object.assign(this.canvasContainer.style,{width:`${raw_width}px`,height:`${raw_height}px`,left:`${this.pan_offset.x}px`,top:`${this.pan_offset.y}px`})}handlePanStart(event){let coords_canvas=this.messageBroker.pull("screenToCanvas",{x:event.offsetX,y:event.offsetY});this.mouseDownPoint={x:event.clientX,y:event.clientY},this.messageBroker.publish("panCursor",!0),this.initialPan=this.pan_offset}handlePanMove(event){if(this.mouseDownPoint===null)throw new Error("mouseDownPoint is null");let deltaX=this.mouseDownPoint.x-event.clientX,deltaY=this.mouseDownPoint.y-event.clientY,pan_x=this.initialPan.x-deltaX,pan_y=this.initialPan.y-deltaY;this.pan_offset={x:pan_x,y:pan_y},this.invalidatePanZoom()}}class MessageBroker{static{__name(this,"MessageBroker")}pushTopics={};pullTopics={};constructor(){this.registerListeners()}registerListeners(){this.createPushTopic("panStart"),this.createPushTopic("paintBucketFill"),this.createPushTopic("saveState"),this.createPushTopic("brushAdjustmentStart"),this.createPushTopic("drawStart"),this.createPushTopic("panMove"),this.createPushTopic("updateBrushPreview"),this.createPushTopic("brushAdjustment"),this.createPushTopic("draw"),this.createPushTopic("paintBucketCursor"),this.createPushTopic("panCursor"),this.createPushTopic("drawEnd"),this.createPushTopic("zoom"),this.createPushTopic("undo"),this.createPushTopic("redo"),this.createPushTopic("cursorPoint"),this.createPushTopic("panOffset"),this.createPushTopic("zoomRatio"),this.createPushTopic("getMaskCanvas"),this.createPushTopic("getCanvasContainer"),this.createPushTopic("screenToCanvas"),this.createPushTopic("isKeyPressed"),this.createPushTopic("isCombinationPressed"),this.createPushTopic("setPaintBucketTolerance"),this.createPushTopic("setBrushSize"),this.createPushTopic("setBrushHardness"),this.createPushTopic("setBrushOpacity"),this.createPushTopic("setBrushShape"),this.createPushTopic("initZoomPan"),this.createPushTopic("setTool"),this.createPushTopic("pointerDown"),this.createPushTopic("pointerMove"),this.createPushTopic("pointerUp"),this.createPushTopic("wheel"),this.createPushTopic("initPaintBucketTool"),this.createPushTopic("setBrushVisibility"),this.createPushTopic("setBrushPreviewGradientVisibility"),this.createPushTopic("handleTouchStart"),this.createPushTopic("handleTouchMove"),this.createPushTopic("handleTouchEnd"),this.createPushTopic("colorSelectFill"),this.createPushTopic("setColorSelectTolerance"),this.createPushTopic("setLivePreview"),this.createPushTopic("updateCursor"),this.createPushTopic("setColorComparisonMethod"),this.createPushTopic("clearLastPoint"),this.createPushTopic("setWholeImage"),this.createPushTopic("setMaskBoundary"),this.createPushTopic("setMaskTolerance"),this.createPushTopic("setBrushSmoothingPrecision"),this.createPushTopic("setZoomText"),this.createPushTopic("resetZoom"),this.createPushTopic("invert")}createPushTopic(topicName){if(this.topicExists(this.pushTopics,topicName))throw new Error("Topic already exists");this.pushTopics[topicName]=[]}subscribe(topicName,callback){if(!this.topicExists(this.pushTopics,topicName))throw new Error(`Topic "${topicName}" does not exist!`);this.pushTopics[topicName].push(callback)}unsubscribe(topicName,callback){if(!this.topicExists(this.pushTopics,topicName))throw new Error("Topic does not exist");const index=this.pushTopics[topicName].indexOf(callback);index>-1&&this.pushTopics[topicName].splice(index,1)}publish(topicName,...args){if(!this.topicExists(this.pushTopics,topicName))throw new Error(`Topic "${topicName}" does not exist!`);this.pushTopics[topicName].forEach(callback=>{callback(...args)})}createPullTopic(topicName,callBack){if(this.topicExists(this.pullTopics,topicName))throw new Error("Topic already exists");this.pullTopics[topicName]=callBack}async pull(topicName,data){if(!this.topicExists(this.pullTopics,topicName))throw new Error("Topic does not exist");const callBack=this.pullTopics[topicName];try{return await callBack(data)}catch(error){throw console.error(`Error pulling data from topic "${topicName}":`,error),error}}topicExists(topics,topicName){return topics.hasOwnProperty(topicName)}}class KeyboardManager{static{__name(this,"KeyboardManager")}keysDown=[];maskEditor;messageBroker;constructor(maskEditor){this.maskEditor=maskEditor,this.messageBroker=maskEditor.getMessageBroker(),this.addPullTopics()}addPullTopics(){this.messageBroker.createPullTopic("isKeyPressed",key=>Promise.resolve(this.isKeyDown(key)))}addListeners(){document.addEventListener("keydown",event=>this.handleKeyDown(event)),document.addEventListener("keyup",event=>this.handleKeyUp(event)),window.addEventListener("blur",()=>this.clearKeys())}removeListeners(){document.removeEventListener("keydown",event=>this.handleKeyDown(event)),document.removeEventListener("keyup",event=>this.handleKeyUp(event))}clearKeys(){this.keysDown=[]}handleKeyDown(event){this.keysDown.includes(event.key)||this.keysDown.push(event.key)}handleKeyUp(event){this.keysDown=this.keysDown.filter(key=>key!==event.key)}isKeyDown(key){return this.keysDown.includes(key)}undoCombinationPressed(){const combination=["ctrl","z"],keysDownLower=this.keysDown.map(key=>key.toLowerCase()),result=combination.every(key=>keysDownLower.includes(key));return result&&this.messageBroker.publish("undo"),result}redoCombinationPressed(){const combination=["ctrl","shift","z"],keysDownLower=this.keysDown.map(key=>key.toLowerCase()),result=combination.every(key=>keysDownLower.includes(key));return result&&this.messageBroker.publish("redo"),result}}app.registerExtension({name:"Comfy.MaskEditor",settings:[{id:"Comfy.MaskEditor.UseNewEditor",category:["Mask Editor","NewEditor"],name:"Use new mask editor",tooltip:"Switch to the new mask editor interface",type:"boolean",defaultValue:!0,experimental:!0},{id:"Comfy.MaskEditor.BrushAdjustmentSpeed",category:["Mask Editor","BrushAdjustment","Sensitivity"],name:"Brush adjustment speed multiplier",tooltip:"Controls how quickly the brush size and hardness change when adjusting. Higher values mean faster changes.",experimental:!0,type:"slider",attrs:{min:.1,max:2,step:.1},defaultValue:1,versionAdded:"1.0.0"},{id:"Comfy.MaskEditor.UseDominantAxis",category:["Mask Editor","BrushAdjustment","UseDominantAxis"],name:"Lock brush adjustment to dominant axis",tooltip:"When enabled, brush adjustments will only affect size OR hardness based on which direction you move more",type:"boolean",defaultValue:!0,experimental:!0}],init(app2){function openMaskEditor(){if(app2.extensionManager.setting.get("Comfy.MaskEditor.UseNewEditor")){const dlg=MaskEditorDialog.getInstance();dlg?.isOpened&&!dlg.isOpened()&&dlg.show()}else{const dlg=MaskEditorDialogOld.getInstance();dlg?.isOpened&&!dlg.isOpened()&&dlg.show()}}__name(openMaskEditor,"openMaskEditor"),ComfyApp.open_maskeditor=openMaskEditor;const context_predicate=__name(()=>!!(ComfyApp.clipspace&&ComfyApp.clipspace.imgs&&ComfyApp.clipspace.imgs.length>0),"context_predicate");ClipspaceDialog.registerButton("MaskEditor",context_predicate,openMaskEditor)}});const id="Comfy.NodeTemplates",file="comfy.templates.json";class ManageTemplates extends ComfyDialog{static{__name(this,"ManageTemplates")}templates;draggedEl;saveVisualCue;emptyImg;importInput;constructor(){super(),this.load().then(v=>{this.templates=v}),this.element.classList.add("comfy-manage-templates"),this.draggedEl=null,this.saveVisualCue=null,this.emptyImg=new Image,this.emptyImg.src="data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=",this.importInput=$el("input",{type:"file",accept:".json",multiple:!0,style:{display:"none"},parent:document.body,onchange:__name(()=>this.importAll(),"onchange")})}createButtons(){const btns=super.createButtons();return btns[0].textContent="Close",btns[0].onclick=e=>{clearTimeout(this.saveVisualCue),this.close()},btns.unshift($el("button",{type:"button",textContent:"Export",onclick:__name(()=>this.exportAll(),"onclick")})),btns.unshift($el("button",{type:"button",textContent:"Import",onclick:__name(()=>{this.importInput.click()},"onclick")})),btns}async load(){let templates=[];const res=await api.getUserData(file);if(res.status===200)try{templates=await res.json()}catch{}else res.status!==404&&console.error(res.status+" "+res.statusText);return templates??[]}async store(){const templates=JSON.stringify(this.templates,void 0,4);try{await api.storeUserData(file,templates,{stringify:!1})}catch(error){console.error(error),useToastStore().addAlert(error.message)}}async importAll(){for(const file2 of this.importInput.files)if(file2.type==="application/json"||file2.name.endsWith(".json")){const reader=new FileReader;reader.onload=async()=>{const importFile=JSON.parse(reader.result);if(importFile?.templates){for(const template of importFile.templates)template?.name&&template?.data&&this.templates.push(template);await this.store()}},await reader.readAsText(file2)}this.importInput.value=null,this.close()}exportAll(){if(this.templates.length==0){useToastStore().addAlert("No templates to export.");return}const json=JSON.stringify({templates:this.templates},null,2),blob=new Blob([json],{type:"application/json"}),url=URL.createObjectURL(blob),a=$el("a",{href:url,download:"node_templates.json",style:{display:"none"},parent:document.body});a.click(),setTimeout(function(){a.remove(),window.URL.revokeObjectURL(url)},0)}show(){super.show($el("div",{},this.templates.flatMap((t2,i)=>{let nameInput;return[$el("div",{dataset:{id:i.toString()},className:"templateManagerRow",style:{display:"grid",gridTemplateColumns:"1fr auto",border:"1px dashed transparent",gap:"5px",backgroundColor:"var(--comfy-menu-bg)"},ondragstart:__name(e=>{this.draggedEl=e.currentTarget,e.currentTarget.style.opacity="0.6",e.currentTarget.style.border="1px dashed yellow",e.dataTransfer.effectAllowed="move",e.dataTransfer.setDragImage(this.emptyImg,0,0)},"ondragstart"),ondragend:__name(e=>{e.target.style.opacity="1",e.currentTarget.style.border="1px dashed transparent",e.currentTarget.removeAttribute("draggable"),this.element.querySelectorAll(".templateManagerRow").forEach((el,i2)=>{var prev_i=Number.parseInt(el.dataset.id);el==this.draggedEl&&prev_i!=i2&&this.templates.splice(i2,0,this.templates.splice(prev_i,1)[0]),el.dataset.id=i2.toString()}),this.store()},"ondragend"),ondragover:__name(e=>{if(e.preventDefault(),e.currentTarget==this.draggedEl)return;let rect=e.currentTarget.getBoundingClientRect();e.clientY>rect.top+rect.height/2?e.currentTarget.parentNode.insertBefore(this.draggedEl,e.currentTarget.nextSibling):e.currentTarget.parentNode.insertBefore(this.draggedEl,e.currentTarget)},"ondragover")},[$el("label",{textContent:"Name: ",style:{cursor:"grab"},onmousedown:__name(e=>{e.target.localName=="label"&&(e.currentTarget.parentNode.draggable="true")},"onmousedown")},[$el("input",{value:t2.name,dataset:{name:t2.name},style:{transitionProperty:"background-color",transitionDuration:"0s"},onchange:__name(e=>{clearTimeout(this.saveVisualCue);var el=e.target,row=el.parentNode.parentNode;this.templates[row.dataset.id].name=el.value.trim()||"untitled",this.store(),el.style.backgroundColor="rgb(40, 95, 40)",el.style.transitionDuration="0s",this.saveVisualCue=setTimeout(function(){el.style.transitionDuration=".7s",el.style.backgroundColor="var(--comfy-input-bg)"},15)},"onchange"),onkeypress:__name(e=>{var el=e.target;clearTimeout(this.saveVisualCue),el.style.transitionDuration="0s",el.style.backgroundColor="var(--comfy-input-bg)"},"onkeypress"),$:__name(el=>nameInput=el,"$")})]),$el("div",{},[$el("button",{textContent:"Export",style:{fontSize:"12px",fontWeight:"normal"},onclick:__name(e=>{const json=JSON.stringify({templates:[t2]},null,2),blob=new Blob([json],{type:"application/json"}),url=URL.createObjectURL(blob),a=$el("a",{href:url,download:(nameInput.value||t2.name)+".json",style:{display:"none"},parent:document.body});a.click(),setTimeout(function(){a.remove(),window.URL.revokeObjectURL(url)},0)},"onclick")}),$el("button",{textContent:"Delete",style:{fontSize:"12px",color:"red",fontWeight:"normal"},onclick:__name(e=>{const item=e.target.parentNode.parentNode;item.parentNode.removeChild(item),this.templates.splice(item.dataset.id*1,1),this.store();var that=this;setTimeout(function(){that.element.querySelectorAll(".templateManagerRow").forEach((el,i2)=>{el.dataset.id=i2.toString()})},0)},"onclick")})])])]})))}}app.registerExtension({name:id,setup(){const manage=new ManageTemplates,clipboardAction=__name(async cb=>{const old=localStorage.getItem("litegrapheditor_clipboard");await cb(),localStorage.setItem("litegrapheditor_clipboard",old)},"clipboardAction"),orig=LGraphCanvas.prototype.getCanvasMenuOptions;LGraphCanvas.prototype.getCanvasMenuOptions=function(){const options=orig.apply(this,arguments);options.push(null),options.push({content:"Save Selected as Template",disabled:!Object.keys(app.canvas.selected_nodes||{}).length,callback:__name(()=>{const name=prompt("Enter name");name?.trim()&&clipboardAction(()=>{app.canvas.copyToClipboard();let data=localStorage.getItem("litegrapheditor_clipboard");data=JSON.parse(data);const nodeIds=Object.keys(app.canvas.selected_nodes);for(let i=0;i<nodeIds.length;i++){const node=app.graph.getNodeById(nodeIds[i]),nodeData=node?.constructor.nodeData;let groupData=GroupNodeHandler.getGroupData(node);groupData&&(groupData=groupData.nodeData,data.groupNodes||(data.groupNodes={}),data.groupNodes[nodeData.name]=groupData,data.nodes[i].type=nodeData.name)}manage.templates.push({name,data:JSON.stringify(data)}),manage.store()})},"callback")});const subItems=manage.templates.map(t2=>({content:t2.name,callback:__name(()=>{clipboardAction(async()=>{const data=JSON.parse(t2.data);await GroupNodeConfig.registerFromWorkflow(data.groupNodes,{}),data.reroutes?(localStorage.setItem("litegrapheditor_clipboard",t2.data),app.canvas.pasteFromClipboard()):deserialiseAndCreate(t2.data,app.canvas)})},"callback")}));return subItems.push(null,{content:"Manage",callback:__name(()=>manage.show(),"callback")}),options.push({content:"Node Templates",submenu:{options:subItems}}),options}}});app.registerExtension({name:"Comfy.NoteNode",registerCustomNodes(){class NoteNode extends LGraphNode{static{__name(this,"NoteNode")}static category;color=LGraphCanvas.node_colors.yellow.color;bgcolor=LGraphCanvas.node_colors.yellow.bgcolor;groupcolor=LGraphCanvas.node_colors.yellow.groupcolor;isVirtualNode;collapsable;title_mode;constructor(title){super(title),this.properties||(this.properties={text:""}),ComfyWidgets.STRING(this,"",["",{default:this.properties.text,multiline:!0}],app),this.serialize_widgets=!0,this.isVirtualNode=!0}}LiteGraph.registerNodeType("Note",Object.assign(NoteNode,{title_mode:LiteGraph.NORMAL_TITLE,title:"Note",collapsable:!0})),NoteNode.category="utils"}});app.registerExtension({name:"Comfy.RerouteNode",registerCustomNodes(app2){class RerouteNode extends LGraphNode{static{__name(this,"RerouteNode")}static category;static defaultVisibility=!1;constructor(title){super(title),this.properties||(this.properties={}),this.properties.showOutputText=RerouteNode.defaultVisibility,this.properties.horizontal=!1,this.addInput("","*"),this.addOutput(this.properties.showOutputText?"*":"","*"),this.onAfterGraphConfigured=function(){requestAnimationFrame(()=>{this.onConnectionsChange(LiteGraph.INPUT,null,!0,null)})},this.onConnectionsChange=(type,index,connected,link_info)=>{if(this.applyOrientation(),connected&&type===LiteGraph.OUTPUT&&new Set(this.outputs[0].links.map(l=>app2.graph.links[l].type).filter(t2=>t2!=="*")).size>1){const linksToDisconnect=[];for(let i=0;i<this.outputs[0].links.length-1;i++){const linkId=this.outputs[0].links[i],link=app2.graph.links[linkId];linksToDisconnect.push(link)}for(const link of linksToDisconnect)app2.graph.getNodeById(link.target_id).disconnectInput(link.target_slot)}let currentNode=this,updateNodes=[],inputType=null,inputNode=null;for(;currentNode;){updateNodes.unshift(currentNode);const linkId=currentNode.inputs[0].link;if(linkId!==null){const link=app2.graph.links[linkId];if(!link)return;const node=app2.graph.getNodeById(link.origin_id);if(node.constructor.type==="Reroute")node===this?(currentNode.disconnectInput(link.target_slot),currentNode=null):currentNode=node;else{inputNode=currentNode,inputType=node.outputs[link.origin_slot]?.type??null;break}}else{currentNode=null;break}}const nodes=[this];let outputType=null;for(;nodes.length;){currentNode=nodes.pop();const outputs=(currentNode.outputs?currentNode.outputs[0].links:[])||[];if(outputs.length)for(const linkId of outputs){const link=app2.graph.links[linkId];if(!link)continue;const node=app2.graph.getNodeById(link.target_id);if(node.constructor.type==="Reroute")nodes.push(node),updateNodes.push(node);else{const nodeOutType=node.inputs&&node.inputs[link?.target_slot]&&node.inputs[link.target_slot].type?node.inputs[link.target_slot].type:null;inputType&&!LiteGraph.isValidConnection(inputType,nodeOutType)?node.disconnectInput(link.target_slot):outputType=nodeOutType}}}const displayType=inputType||outputType||"*",color=LGraphCanvas.link_type_colors[displayType];let widgetConfig,targetWidget,widgetType;for(const node of updateNodes){node.outputs[0].type=inputType||"*",node.__outputType=displayType,node.outputs[0].name=node.properties.showOutputText?displayType:"",node.size=node.computeSize(),node.applyOrientation();for(const l of node.outputs[0].links||[]){const link=app2.graph.links[l];if(link){if(link.color=color,app2.configuringGraph)continue;const targetNode=app2.graph.getNodeById(link.target_id),targetInput=targetNode.inputs?.[link.target_slot];if(targetInput?.widget){const config=getWidgetConfig(targetInput);widgetConfig||(widgetConfig=config[1]??{},widgetType=config[0]),targetWidget||(targetWidget=targetNode.widgets?.find(w=>w.name===targetInput.widget.name));const merged=mergeIfValid(targetInput,[config[0],widgetConfig]);merged.customConfig&&(widgetConfig=merged.customConfig)}}}}for(const node of updateNodes)widgetConfig&&outputType?(node.inputs[0].widget={name:"value"},setWidgetConfig(node.inputs[0],[widgetType??displayType,widgetConfig],targetWidget)):setWidgetConfig(node.inputs[0],null);if(inputNode){const link=app2.graph.links[inputNode.inputs[0].link];link&&(link.color=color)}},this.clone=function(){const cloned=RerouteNode.prototype.clone.apply(this);return cloned.removeOutput(0),cloned.addOutput(this.properties.showOutputText?"*":"","*"),cloned.size=cloned.computeSize(),cloned},this.isVirtualNode=!0}getExtraMenuOptions(_,options){return options.unshift({content:(this.properties.showOutputText?"Hide":"Show")+" Type",callback:__name(()=>{this.properties.showOutputText=!this.properties.showOutputText,this.properties.showOutputText?this.outputs[0].name=this.__outputType||this.outputs[0].type:this.outputs[0].name="",this.size=this.computeSize(),this.applyOrientation(),app2.graph.setDirtyCanvas(!0,!0)},"callback")},{content:(RerouteNode.defaultVisibility?"Hide":"Show")+" Type By Default",callback:__name(()=>{RerouteNode.setDefaultTextVisibility(!RerouteNode.defaultVisibility)},"callback")},{content:"Set "+(this.properties.horizontal?"Horizontal":"Vertical"),callback:__name(()=>{this.properties.horizontal=!this.properties.horizontal,this.applyOrientation()},"callback")}),[]}applyOrientation(){this.horizontal=this.properties.horizontal,this.horizontal?this.inputs[0].pos=[this.size[0]/2,0]:delete this.inputs[0].pos,app2.graph.setDirtyCanvas(!0,!0)}computeSize(){return[this.properties.showOutputText&&this.outputs&&this.outputs.length?Math.max(75,LiteGraph.NODE_TEXT_SIZE*this.outputs[0].name.length*.6+40):75,26]}static setDefaultTextVisibility(visible){RerouteNode.defaultVisibility=visible,visible?localStorage["Comfy.RerouteNode.DefaultVisibility"]="true":delete localStorage["Comfy.RerouteNode.DefaultVisibility"]}}RerouteNode.setDefaultTextVisibility(!!localStorage["Comfy.RerouteNode.DefaultVisibility"]),LiteGraph.registerNodeType("Reroute",Object.assign(RerouteNode,{title_mode:LiteGraph.NO_TITLE,title:"Reroute",collapsable:!1})),RerouteNode.category="utils"}});app.registerExtension({name:"Comfy.SaveImageExtraOutput",async beforeRegisterNodeDef(nodeType,nodeData,app2){if(nodeData.name==="SaveImage"||nodeData.name==="SaveAnimatedWEBP"){const onNodeCreated=nodeType.prototype.onNodeCreated;nodeType.prototype.onNodeCreated=function(){const r=onNodeCreated?onNodeCreated.apply(this,arguments):void 0,widget=this.widgets.find(w=>w.name==="filename_prefix");return widget.serializeValue=()=>applyTextReplacements(app2,widget.value),r}}else{const onNodeCreated=nodeType.prototype.onNodeCreated;nodeType.prototype.onNodeCreated=function(){const r=onNodeCreated?onNodeCreated.apply(this,arguments):void 0;return(!this.properties||!("Node name for S&R"in this.properties))&&this.addProperty("Node name for S&R",this.constructor.type,"string"),r}}}});let touchZooming,touchCount=0;app.registerExtension({name:"Comfy.SimpleTouchSupport",setup(){let touchDist,touchTime,lastTouch,lastScale;function getMultiTouchPos(e){return Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY)}__name(getMultiTouchPos,"getMultiTouchPos");function getMultiTouchCenter(e){return{clientX:(e.touches[0].clientX+e.touches[1].clientX)/2,clientY:(e.touches[0].clientY+e.touches[1].clientY)/2}}__name(getMultiTouchCenter,"getMultiTouchCenter"),app.canvasEl.parentElement.addEventListener("touchstart",e=>{touchCount++,lastTouch=null,lastScale=null,e.touches?.length===1?(touchTime=new Date,lastTouch=e.touches[0]):(touchTime=null,e.touches?.length===2&&(lastScale=app.canvas.ds.scale,lastTouch=getMultiTouchCenter(e),touchDist=getMultiTouchPos(e),app.canvas.pointer.isDown=!1))},!0),app.canvasEl.parentElement.addEventListener("touchend",e=>{touchCount--,e.touches?.length!==1&&(touchZooming=!1),touchTime&&!e.touches?.length&&(new Date().getTime()-touchTime>600&&e.target===app.canvasEl&&(app.canvasEl.dispatchEvent(new PointerEvent("pointerdown",{button:2,clientX:e.changedTouches[0].clientX,clientY:e.changedTouches[0].clientY})),e.preventDefault()),touchTime=null)}),app.canvasEl.parentElement.addEventListener("touchmove",e=>{if(touchTime=null,e.touches?.length===2&&lastTouch&&!e.ctrlKey&&!e.shiftKey){e.preventDefault(),app.canvas.pointer.isDown=!1,touchZooming=!0,LiteGraph.closeAllContextMenus(window),app.canvas.search_box?.close();const newTouchDist=getMultiTouchPos(e),center=getMultiTouchCenter(e);let scale=lastScale*newTouchDist/touchDist;const newX=(center.clientX-lastTouch.clientX)/scale,newY=(center.clientY-lastTouch.clientY)/scale;scale<app.canvas.ds.min_scale?scale=app.canvas.ds.min_scale:scale>app.canvas.ds.max_scale&&(scale=app.canvas.ds.max_scale);const oldScale=app.canvas.ds.scale;app.canvas.ds.scale=scale,Math.abs(app.canvas.ds.scale-1)<.01&&(app.canvas.ds.scale=1);const newScale=app.canvas.ds.scale,convertScaleToOffset=__name(scale2=>[center.clientX/scale2-app.canvas.ds.offset[0],center.clientY/scale2-app.canvas.ds.offset[1]],"convertScaleToOffset");var oldCenter=convertScaleToOffset(oldScale),newCenter=convertScaleToOffset(newScale);app.canvas.ds.offset[0]+=newX+newCenter[0]-oldCenter[0],app.canvas.ds.offset[1]+=newY+newCenter[1]-oldCenter[1],lastTouch.clientX=center.clientX,lastTouch.clientY=center.clientY,app.canvas.setDirty(!0,!0)}},!0)}});const processMouseDown=LGraphCanvas.prototype.processMouseDown;LGraphCanvas.prototype.processMouseDown=function(e){if(!(touchZooming||touchCount))return app.canvas.pointer.isDown=!1,processMouseDown.apply(this,arguments)};const processMouseMove=LGraphCanvas.prototype.processMouseMove;LGraphCanvas.prototype.processMouseMove=function(e){if(!(touchZooming||touchCount>1))return processMouseMove.apply(this,arguments)};app.registerExtension({name:"Comfy.SlotDefaults",suggestionsNumber:null,init(){LiteGraph.search_filter_enabled=!0,LiteGraph.middle_click_slot_add_default_node=!0,this.suggestionsNumber=app.ui.settings.addSetting({id:"Comfy.NodeSuggestions.number",category:["Comfy","Node Search Box","NodeSuggestions"],name:"Number of nodes suggestions",tooltip:"Only for litegraph searchbox/context menu",type:"slider",attrs:{min:1,max:100,step:1},defaultValue:5,onChange:__name((newVal,oldVal)=>{this.setDefaults(newVal)},"onChange")})},slot_types_default_out:{},slot_types_default_in:{},async beforeRegisterNodeDef(nodeType,nodeData,app2){var nodeId=nodeData.name;const inputs=nodeData.input?.required;for(const inputKey in inputs){var input=inputs[inputKey];if(typeof input[0]!="string")continue;var type=input[0];if(type in ComfyWidgets){var customProperties=input[1];if(!customProperties?.forceInput)continue}if(type in this.slot_types_default_out||(this.slot_types_default_out[type]=["Reroute"]),this.slot_types_default_out[type].includes(nodeId))continue;this.slot_types_default_out[type].push(nodeId);const lowerType=type.toLocaleLowerCase();lowerType in LiteGraph.registered_slot_in_types||(LiteGraph.registered_slot_in_types[lowerType]={nodes:[]}),LiteGraph.registered_slot_in_types[lowerType].nodes.push(nodeType.comfyClass)}var outputs=nodeData.output??[];for(const el of outputs){const type2=el;type2 in this.slot_types_default_in||(this.slot_types_default_in[type2]=["Reroute"]),this.slot_types_default_in[type2].push(nodeId),type2 in LiteGraph.registered_slot_out_types||(LiteGraph.registered_slot_out_types[type2]={nodes:[]}),LiteGraph.registered_slot_out_types[type2].nodes.push(nodeType.comfyClass),LiteGraph.slot_types_out.includes(type2)||LiteGraph.slot_types_out.push(type2)}var maxNum=this.suggestionsNumber.value;this.setDefaults(maxNum)},setDefaults(maxNum){LiteGraph.slot_types_default_out={},LiteGraph.slot_types_default_in={};for(const type in this.slot_types_default_out)LiteGraph.slot_types_default_out[type]=this.slot_types_default_out[type].slice(0,maxNum);for(const type in this.slot_types_default_in)LiteGraph.slot_types_default_in[type]=this.slot_types_default_in[type].slice(0,maxNum)}});app.registerExtension({name:"Comfy.UploadImage",beforeRegisterNodeDef(nodeType,nodeData){nodeData?.input?.required?.image?.[1]?.image_upload===!0&&(nodeData.input.required.upload=["IMAGEUPLOAD"])}});const WEBCAM_READY=Symbol();app.registerExtension({name:"Comfy.WebcamCapture",getCustomWidgets(app2){return{WEBCAM(node,inputName){let res;node[WEBCAM_READY]=new Promise(resolve=>res=resolve);const container=document.createElement("div");container.style.background="rgba(0,0,0,0.25)",container.style.textAlign="center";const video=document.createElement("video");return video.style.height=video.style.width="100%",__name(async()=>{try{const stream=await navigator.mediaDevices.getUserMedia({video:!0,audio:!1});container.replaceChildren(video),setTimeout(()=>res(video),500),video.addEventListener("loadedmetadata",()=>res(video),!1),video.srcObject=stream,video.play()}catch(error){const label=document.createElement("div");label.style.color="red",label.style.overflow="auto",label.style.maxHeight="100%",label.style.whiteSpace="pre-wrap",window.isSecureContext?label.textContent=`Unable to load webcam, please ensure access is granted:
`+error.message:label.textContent=`Unable to load webcam. A secure context is required, if you are not accessing ComfyUI on localhost (127.0.0.1) you will have to enable TLS (https)

`+error.message,container.replaceChildren(label)}},"loadVideo")(),{widget:node.addDOMWidget(inputName,"WEBCAM",container)}}}},nodeCreated(node){if(node.type,node.constructor.comfyClass!=="WebcamCapture")return;let video;const camera=node.widgets.find(w2=>w2.name==="image"),w=node.widgets.find(w2=>w2.name==="width"),h=node.widgets.find(w2=>w2.name==="height"),captureOnQueue=node.widgets.find(w2=>w2.name==="capture_on_queue"),canvas=document.createElement("canvas"),capture=__name(()=>{canvas.width=w.value,canvas.height=h.value,canvas.getContext("2d").drawImage(video,0,0,w.value,h.value);const data=canvas.toDataURL("image/png"),img=new Image;img.onload=()=>{node.imgs=[img],app.graph.setDirtyCanvas(!0),requestAnimationFrame(()=>{node.setSizeForImage?.()})},img.src=data},"capture"),btn=node.addWidget("button","waiting for camera...","capture",capture);btn.disabled=!0,btn.serializeValue=()=>{},camera.serializeValue=async()=>{if(captureOnQueue.value)capture();else if(!node.imgs?.length){const err2="No webcam image captured";throw useToastStore().addAlert(err2),new Error(err2)}const blob=await new Promise(r=>canvas.toBlob(r)),name=`${+new Date}.png`,file2=new File([blob],name),body=new FormData;body.append("image",file2),body.append("subfolder","webcam"),body.append("type","temp");const resp=await api.fetchApi("/upload/image",{method:"POST",body});if(resp.status!==200){const err2=`Error uploading camera image: ${resp.status} - ${resp.statusText}`;throw useToastStore().addAlert(err2),new Error(err2)}return`webcam/${name} [temp]`},node[WEBCAM_READY].then(v=>{video=v,w.value||(w.value=video.videoWidth||640,h.value=video.videoHeight||480),btn.disabled=!1,btn.label="capture"})}});function splitFilePath$1(path){const folder_separator=path.lastIndexOf("/");return folder_separator===-1?["",path]:[path.substring(0,folder_separator),path.substring(folder_separator+1)]}__name(splitFilePath$1,"splitFilePath$1");function getResourceURL$1(subfolder,filename,type="input"){return`/view?${["filename="+encodeURIComponent(filename),"type="+type,"subfolder="+subfolder,app.getRandParam().substring(1)].join("&")}`}__name(getResourceURL$1,"getResourceURL$1");async function uploadFile$1(audioWidget,audioUIWidget,file2,updateNode,pasted=!1){try{const body=new FormData;body.append("image",file2),pasted&&body.append("subfolder","pasted");const resp=await api.fetchApi("/upload/image",{method:"POST",body});if(resp.status===200){const data=await resp.json();let path=data.name;data.subfolder&&(path=data.subfolder+"/"+path),audioWidget.options.values.includes(path)||audioWidget.options.values.push(path),updateNode&&(audioUIWidget.element.src=api.apiURL(getResourceURL$1(...splitFilePath$1(path))),audioWidget.value=path)}else useToastStore().addAlert(resp.status+" - "+resp.statusText)}catch(error){useToastStore().addAlert(error)}}__name(uploadFile$1,"uploadFile$1");app.registerExtension({name:"Comfy.AudioWidget",async beforeRegisterNodeDef(nodeType,nodeData){["LoadAudio","SaveAudio","PreviewAudio"].includes(nodeType.comfyClass)&&(nodeData.input.required.audioUI=["AUDIO_UI"])},getCustomWidgets(){return{AUDIO_UI(node,inputName){const audio=document.createElement("audio");audio.controls=!0,audio.classList.add("comfy-audio"),audio.setAttribute("name","media");const audioUIWidget=node.addDOMWidget(inputName,"audioUI",audio,{serialize:!1});if(node.constructor.nodeData.output_node){audioUIWidget.element.classList.add("empty-audio-widget");const onExecuted=node.onExecuted;node.onExecuted=function(message){onExecuted?.apply(this,arguments);const audios=message.audio;if(!audios)return;const audio2=audios[0];audioUIWidget.element.src=api.apiURL(getResourceURL$1(audio2.subfolder,audio2.filename,audio2.type)),audioUIWidget.element.classList.remove("empty-audio-widget")}}return{widget:audioUIWidget}}}},onNodeOutputsUpdated(nodeOutputs){for(const[nodeId,output]of Object.entries(nodeOutputs)){const node=app.graph.getNodeById(nodeId);if("audio"in output){const audioUIWidget=node.widgets.find(w=>w.name==="audioUI"),audio=output.audio[0];audioUIWidget.element.src=api.apiURL(getResourceURL$1(audio.subfolder,audio.filename,audio.type)),audioUIWidget.element.classList.remove("empty-audio-widget")}}}});app.registerExtension({name:"Comfy.UploadAudio",async beforeRegisterNodeDef(nodeType,nodeData){nodeData?.input?.required?.audio?.[1]?.audio_upload===!0&&(nodeData.input.required.upload=["AUDIOUPLOAD"])},getCustomWidgets(){return{AUDIOUPLOAD(node,inputName){const audioWidget=node.widgets.find(w=>w.name==="audio"),audioUIWidget=node.widgets.find(w=>w.name==="audioUI"),onAudioWidgetUpdate=__name(()=>{audioUIWidget.element.src=api.apiURL(getResourceURL$1(...splitFilePath$1(audioWidget.value)))},"onAudioWidgetUpdate");audioWidget.value&&onAudioWidgetUpdate(),audioWidget.callback=onAudioWidgetUpdate;const onGraphConfigured=node.onGraphConfigured;node.onGraphConfigured=function(){onGraphConfigured?.apply(this,arguments),audioWidget.value&&onAudioWidgetUpdate()};const fileInput=document.createElement("input");fileInput.type="file",fileInput.accept="audio/*",fileInput.style.display="none",fileInput.onchange=()=>{fileInput.files.length&&uploadFile$1(audioWidget,audioUIWidget,fileInput.files[0],!0)};const uploadWidget=node.addWidget("button",inputName,"",()=>{fileInput.click()},{serialize:!1});return uploadWidget.label="choose file to upload",{widget:uploadWidget}}}}});(async()=>{if(!isElectron())return;const electronAPI$1=electronAPI(),desktopAppVersion=await electronAPI$1.getElectronVersion(),onChangeRestartApp=__name((newValue,oldValue)=>{oldValue!==void 0&&newValue!==oldValue&&electronAPI$1.restartApp("Restart ComfyUI to apply changes.",1500)},"onChangeRestartApp");app.registerExtension({name:"Comfy.ElectronAdapter",settings:[{id:"Comfy-Desktop.AutoUpdate",category:["Comfy-Desktop","General","AutoUpdate"],name:"Automatically check for updates",type:"boolean",defaultValue:!0,onChange:onChangeRestartApp},{id:"Comfy-Desktop.SendStatistics",category:["Comfy-Desktop","General","Send Statistics"],name:"Send anonymous crash reports",type:"boolean",defaultValue:!0,onChange:onChangeRestartApp}],commands:[{id:"Comfy-Desktop.Folders.OpenLogsFolder",label:"Open Logs Folder",icon:"pi pi-folder-open",function(){electronAPI$1.openLogsFolder()}},{id:"Comfy-Desktop.Folders.OpenModelsFolder",label:"Open Models Folder",icon:"pi pi-folder-open",function(){electronAPI$1.openModelsFolder()}},{id:"Comfy-Desktop.Folders.OpenOutputsFolder",label:"Open Outputs Folder",icon:"pi pi-folder-open",function(){electronAPI$1.openOutputsFolder()}},{id:"Comfy-Desktop.Folders.OpenInputsFolder",label:"Open Inputs Folder",icon:"pi pi-folder-open",function(){electronAPI$1.openInputsFolder()}},{id:"Comfy-Desktop.Folders.OpenCustomNodesFolder",label:"Open Custom Nodes Folder",icon:"pi pi-folder-open",function(){electronAPI$1.openCustomNodesFolder()}},{id:"Comfy-Desktop.Folders.OpenModelConfig",label:"Open extra_model_paths.yaml",icon:"pi pi-file",function(){electronAPI$1.openModelConfig()}},{id:"Comfy-Desktop.OpenDevTools",label:"Open DevTools",icon:"pi pi-code",function(){electronAPI$1.openDevTools()}},{id:"Comfy-Desktop.OpenFeedbackPage",label:"Feedback",icon:"pi pi-envelope",function(){window.open("https://forum.comfy.org/c/v1-feedback/","_blank")}},{id:"Comfy-Desktop.Reinstall",label:t("desktopMenu.reinstall"),icon:"pi pi-refresh",async function(){await showConfirmationDialog({message:t("desktopMenu.confirmReinstall"),title:t("desktopMenu.reinstall"),type:"reinstall"})&&electronAPI$1.reinstall()}},{id:"Comfy-Desktop.Restart",label:"Restart",icon:"pi pi-refresh",function(){electronAPI$1.restartApp()}}],menuCommands:[{path:["Help"],commands:["Comfy-Desktop.OpenFeedbackPage"]},{path:["Help"],commands:["Comfy-Desktop.OpenDevTools"]},{path:["Help","Open Folder"],commands:["Comfy-Desktop.Folders.OpenLogsFolder","Comfy-Desktop.Folders.OpenModelsFolder","Comfy-Desktop.Folders.OpenOutputsFolder","Comfy-Desktop.Folders.OpenInputsFolder","Comfy-Desktop.Folders.OpenCustomNodesFolder","Comfy-Desktop.Folders.OpenModelConfig"]},{path:["Help"],commands:["Comfy-Desktop.Reinstall"]}],aboutPageBadges:[{label:"ComfyUI_desktop v"+desktopAppVersion,url:"https://github.com/Comfy-Org/electron",icon:"pi pi-github"}]})})();/**
 * @license
 * Copyright 2010-2024 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const REVISION="170",MOUSE={LEFT:0,MIDDLE:1,RIGHT:2,ROTATE:0,DOLLY:1,PAN:2},TOUCH={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},CullFaceNone=0,CullFaceBack=1,CullFaceFront=2;const PCFShadowMap=1,PCFSoftShadowMap=2,VSMShadowMap=3,FrontSide=0,BackSide=1,DoubleSide=2,NoBlending=0,NormalBlending=1,AdditiveBlending=2,SubtractiveBlending=3,MultiplyBlending=4,CustomBlending=5,AddEquation=100,SubtractEquation=101,ReverseSubtractEquation=102,MinEquation=103,MaxEquation=104,ZeroFactor=200,OneFactor=201,SrcColorFactor=202,OneMinusSrcColorFactor=203,SrcAlphaFactor=204,OneMinusSrcAlphaFactor=205,DstAlphaFactor=206,OneMinusDstAlphaFactor=207,DstColorFactor=208,OneMinusDstColorFactor=209,SrcAlphaSaturateFactor=210,ConstantColorFactor=211,OneMinusConstantColorFactor=212,ConstantAlphaFactor=213,OneMinusConstantAlphaFactor=214,NeverDepth=0,AlwaysDepth=1,LessDepth=2,LessEqualDepth=3,EqualDepth=4,GreaterEqualDepth=5,GreaterDepth=6,NotEqualDepth=7,MultiplyOperation=0,MixOperation=1,AddOperation=2,NoToneMapping=0,LinearToneMapping=1,ReinhardToneMapping=2,CineonToneMapping=3,ACESFilmicToneMapping=4,CustomToneMapping=5,AgXToneMapping=6,NeutralToneMapping=7,AttachedBindMode="attached",DetachedBindMode="detached",UVMapping=300,CubeReflectionMapping=301,CubeRefractionMapping=302,EquirectangularReflectionMapping=303,EquirectangularRefractionMapping=304,CubeUVReflectionMapping=306,RepeatWrapping=1e3,ClampToEdgeWrapping=1001,MirroredRepeatWrapping=1002,NearestFilter=1003,NearestMipmapNearestFilter=1004;const NearestMipmapLinearFilter=1005;const LinearFilter=1006,LinearMipmapNearestFilter=1007;const LinearMipmapLinearFilter=1008;const UnsignedByteType=1009,ByteType=1010,ShortType=1011,UnsignedShortType=1012,IntType=1013,UnsignedIntType=1014,FloatType=1015,HalfFloatType=1016,UnsignedShort4444Type=1017,UnsignedShort5551Type=1018,UnsignedInt248Type=1020,UnsignedInt5999Type=35902,AlphaFormat=1021,RGBFormat=1022,RGBAFormat=1023,LuminanceFormat=1024,LuminanceAlphaFormat=1025,DepthFormat=1026,DepthStencilFormat=1027,RedFormat=1028,RedIntegerFormat=1029,RGFormat=1030,RGIntegerFormat=1031;const RGBAIntegerFormat=1033,RGB_S3TC_DXT1_Format=33776,RGBA_S3TC_DXT1_Format=33777,RGBA_S3TC_DXT3_Format=33778,RGBA_S3TC_DXT5_Format=33779,RGB_PVRTC_4BPPV1_Format=35840,RGB_PVRTC_2BPPV1_Format=35841,RGBA_PVRTC_4BPPV1_Format=35842,RGBA_PVRTC_2BPPV1_Format=35843,RGB_ETC1_Format=36196,RGB_ETC2_Format=37492,RGBA_ETC2_EAC_Format=37496,RGBA_ASTC_4x4_Format=37808,RGBA_ASTC_5x4_Format=37809,RGBA_ASTC_5x5_Format=37810,RGBA_ASTC_6x5_Format=37811,RGBA_ASTC_6x6_Format=37812,RGBA_ASTC_8x5_Format=37813,RGBA_ASTC_8x6_Format=37814,RGBA_ASTC_8x8_Format=37815,RGBA_ASTC_10x5_Format=37816,RGBA_ASTC_10x6_Format=37817,RGBA_ASTC_10x8_Format=37818,RGBA_ASTC_10x10_Format=37819,RGBA_ASTC_12x10_Format=37820,RGBA_ASTC_12x12_Format=37821,RGBA_BPTC_Format=36492,RGB_BPTC_SIGNED_Format=36494,RGB_BPTC_UNSIGNED_Format=36495,RED_RGTC1_Format=36283,SIGNED_RED_RGTC1_Format=36284,RED_GREEN_RGTC2_Format=36285,SIGNED_RED_GREEN_RGTC2_Format=36286,LoopOnce=2200,LoopRepeat=2201,LoopPingPong=2202,InterpolateDiscrete=2300,InterpolateLinear=2301,InterpolateSmooth=2302,ZeroCurvatureEnding=2400,ZeroSlopeEnding=2401,WrapAroundEnding=2402,NormalAnimationBlendMode=2500,AdditiveAnimationBlendMode=2501,TrianglesDrawMode=0,TriangleStripDrawMode=1,TriangleFanDrawMode=2,BasicDepthPacking=3200,RGBADepthPacking=3201;const TangentSpaceNormalMap=0,ObjectSpaceNormalMap=1,NoColorSpace="",SRGBColorSpace="srgb",LinearSRGBColorSpace="srgb-linear",LinearTransfer="linear",SRGBTransfer="srgb";const KeepStencilOp=7680;const AlwaysStencilFunc=519,NeverCompare=512,LessCompare=513,EqualCompare=514,LessEqualCompare=515,GreaterCompare=516,NotEqualCompare=517,GreaterEqualCompare=518,AlwaysCompare=519,StaticDrawUsage=35044;const GLSL3="300 es",WebGLCoordinateSystem=2e3,WebGPUCoordinateSystem=2001;class EventDispatcher{static{__name(this,"EventDispatcher")}addEventListener(type,listener){this._listeners===void 0&&(this._listeners={});const listeners=this._listeners;listeners[type]===void 0&&(listeners[type]=[]),listeners[type].indexOf(listener)===-1&&listeners[type].push(listener)}hasEventListener(type,listener){if(this._listeners===void 0)return!1;const listeners=this._listeners;return listeners[type]!==void 0&&listeners[type].indexOf(listener)!==-1}removeEventListener(type,listener){if(this._listeners===void 0)return;const listenerArray=this._listeners[type];if(listenerArray!==void 0){const index=listenerArray.indexOf(listener);index!==-1&&listenerArray.splice(index,1)}}dispatchEvent(event){if(this._listeners===void 0)return;const listenerArray=this._listeners[event.type];if(listenerArray!==void 0){event.target=this;const array=listenerArray.slice(0);for(let i=0,l=array.length;i<l;i++)array[i].call(this,event);event.target=null}}}const _lut=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];let _seed=1234567;const DEG2RAD=Math.PI/180,RAD2DEG=180/Math.PI;function generateUUID(){const d0=Math.random()*4294967295|0,d1=Math.random()*4294967295|0,d2=Math.random()*4294967295|0,d3=Math.random()*4294967295|0;return(_lut[d0&255]+_lut[d0>>8&255]+_lut[d0>>16&255]+_lut[d0>>24&255]+"-"+_lut[d1&255]+_lut[d1>>8&255]+"-"+_lut[d1>>16&15|64]+_lut[d1>>24&255]+"-"+_lut[d2&63|128]+_lut[d2>>8&255]+"-"+_lut[d2>>16&255]+_lut[d2>>24&255]+_lut[d3&255]+_lut[d3>>8&255]+_lut[d3>>16&255]+_lut[d3>>24&255]).toLowerCase()}__name(generateUUID,"generateUUID");function clamp(value,min,max2){return Math.max(min,Math.min(max2,value))}__name(clamp,"clamp");function euclideanModulo(n,m){return(n%m+m)%m}__name(euclideanModulo,"euclideanModulo");function mapLinear(x,a1,a2,b1,b2){return b1+(x-a1)*(b2-b1)/(a2-a1)}__name(mapLinear,"mapLinear");function inverseLerp(x,y,value){return x!==y?(value-x)/(y-x):0}__name(inverseLerp,"inverseLerp");function lerp(x,y,t2){return(1-t2)*x+t2*y}__name(lerp,"lerp");function damp(x,y,lambda,dt){return lerp(x,y,1-Math.exp(-lambda*dt))}__name(damp,"damp");function pingpong(x,length=1){return length-Math.abs(euclideanModulo(x,length*2)-length)}__name(pingpong,"pingpong");function smoothstep(x,min,max2){return x<=min?0:x>=max2?1:(x=(x-min)/(max2-min),x*x*(3-2*x))}__name(smoothstep,"smoothstep");function smootherstep(x,min,max2){return x<=min?0:x>=max2?1:(x=(x-min)/(max2-min),x*x*x*(x*(x*6-15)+10))}__name(smootherstep,"smootherstep");function randInt(low,high){return low+Math.floor(Math.random()*(high-low+1))}__name(randInt,"randInt");function randFloat(low,high){return low+Math.random()*(high-low)}__name(randFloat,"randFloat");function randFloatSpread(range){return range*(.5-Math.random())}__name(randFloatSpread,"randFloatSpread");function seededRandom(s){s!==void 0&&(_seed=s);let t2=_seed+=1831565813;return t2=Math.imul(t2^t2>>>15,t2|1),t2^=t2+Math.imul(t2^t2>>>7,t2|61),((t2^t2>>>14)>>>0)/4294967296}__name(seededRandom,"seededRandom");function degToRad(degrees){return degrees*DEG2RAD}__name(degToRad,"degToRad");function radToDeg(radians){return radians*RAD2DEG}__name(radToDeg,"radToDeg");function isPowerOfTwo(value){return(value&value-1)===0&&value!==0}__name(isPowerOfTwo,"isPowerOfTwo");function ceilPowerOfTwo(value){return Math.pow(2,Math.ceil(Math.log(value)/Math.LN2))}__name(ceilPowerOfTwo,"ceilPowerOfTwo");function floorPowerOfTwo(value){return Math.pow(2,Math.floor(Math.log(value)/Math.LN2))}__name(floorPowerOfTwo,"floorPowerOfTwo");function setQuaternionFromProperEuler(q,a,b,c,order){const cos=Math.cos,sin=Math.sin,c2=cos(b/2),s2=sin(b/2),c13=cos((a+c)/2),s13=sin((a+c)/2),c1_3=cos((a-c)/2),s1_3=sin((a-c)/2),c3_1=cos((c-a)/2),s3_1=sin((c-a)/2);switch(order){case"XYX":q.set(c2*s13,s2*c1_3,s2*s1_3,c2*c13);break;case"YZY":q.set(s2*s1_3,c2*s13,s2*c1_3,c2*c13);break;case"ZXZ":q.set(s2*c1_3,s2*s1_3,c2*s13,c2*c13);break;case"XZX":q.set(c2*s13,s2*s3_1,s2*c3_1,c2*c13);break;case"YXY":q.set(s2*c3_1,c2*s13,s2*s3_1,c2*c13);break;case"ZYZ":q.set(s2*s3_1,s2*c3_1,c2*s13,c2*c13);break;default:console.warn("THREE.MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+order)}}__name(setQuaternionFromProperEuler,"setQuaternionFromProperEuler");function denormalize(value,array){switch(array.constructor){case Float32Array:return value;case Uint32Array:return value/4294967295;case Uint16Array:return value/65535;case Uint8Array:return value/255;case Int32Array:return Math.max(value/2147483647,-1);case Int16Array:return Math.max(value/32767,-1);case Int8Array:return Math.max(value/127,-1);default:throw new Error("Invalid component type.")}}__name(denormalize,"denormalize");function normalize(value,array){switch(array.constructor){case Float32Array:return value;case Uint32Array:return Math.round(value*4294967295);case Uint16Array:return Math.round(value*65535);case Uint8Array:return Math.round(value*255);case Int32Array:return Math.round(value*2147483647);case Int16Array:return Math.round(value*32767);case Int8Array:return Math.round(value*127);default:throw new Error("Invalid component type.")}}__name(normalize,"normalize");const MathUtils={DEG2RAD,RAD2DEG,generateUUID,clamp,euclideanModulo,mapLinear,inverseLerp,lerp,damp,pingpong,smoothstep,smootherstep,randInt,randFloat,randFloatSpread,seededRandom,degToRad,radToDeg,isPowerOfTwo,ceilPowerOfTwo,floorPowerOfTwo,setQuaternionFromProperEuler,normalize,denormalize};class Vector2{static{__name(this,"Vector2")}constructor(x=0,y=0){Vector2.prototype.isVector2=!0,this.x=x,this.y=y}get width(){return this.x}set width(value){this.x=value}get height(){return this.y}set height(value){this.y=value}set(x,y){return this.x=x,this.y=y,this}setScalar(scalar){return this.x=scalar,this.y=scalar,this}setX(x){return this.x=x,this}setY(y){return this.y=y,this}setComponent(index,value){switch(index){case 0:this.x=value;break;case 1:this.y=value;break;default:throw new Error("index is out of range: "+index)}return this}getComponent(index){switch(index){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+index)}}clone(){return new this.constructor(this.x,this.y)}copy(v){return this.x=v.x,this.y=v.y,this}add(v){return this.x+=v.x,this.y+=v.y,this}addScalar(s){return this.x+=s,this.y+=s,this}addVectors(a,b){return this.x=a.x+b.x,this.y=a.y+b.y,this}addScaledVector(v,s){return this.x+=v.x*s,this.y+=v.y*s,this}sub(v){return this.x-=v.x,this.y-=v.y,this}subScalar(s){return this.x-=s,this.y-=s,this}subVectors(a,b){return this.x=a.x-b.x,this.y=a.y-b.y,this}multiply(v){return this.x*=v.x,this.y*=v.y,this}multiplyScalar(scalar){return this.x*=scalar,this.y*=scalar,this}divide(v){return this.x/=v.x,this.y/=v.y,this}divideScalar(scalar){return this.multiplyScalar(1/scalar)}applyMatrix3(m){const x=this.x,y=this.y,e=m.elements;return this.x=e[0]*x+e[3]*y+e[6],this.y=e[1]*x+e[4]*y+e[7],this}min(v){return this.x=Math.min(this.x,v.x),this.y=Math.min(this.y,v.y),this}max(v){return this.x=Math.max(this.x,v.x),this.y=Math.max(this.y,v.y),this}clamp(min,max2){return this.x=Math.max(min.x,Math.min(max2.x,this.x)),this.y=Math.max(min.y,Math.min(max2.y,this.y)),this}clampScalar(minVal,maxVal){return this.x=Math.max(minVal,Math.min(maxVal,this.x)),this.y=Math.max(minVal,Math.min(maxVal,this.y)),this}clampLength(min,max2){const length=this.length();return this.divideScalar(length||1).multiplyScalar(Math.max(min,Math.min(max2,length)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(v){return this.x*v.x+this.y*v.y}cross(v){return this.x*v.y-this.y*v.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(v){const denominator=Math.sqrt(this.lengthSq()*v.lengthSq());if(denominator===0)return Math.PI/2;const theta=this.dot(v)/denominator;return Math.acos(clamp(theta,-1,1))}distanceTo(v){return Math.sqrt(this.distanceToSquared(v))}distanceToSquared(v){const dx=this.x-v.x,dy=this.y-v.y;return dx*dx+dy*dy}manhattanDistanceTo(v){return Math.abs(this.x-v.x)+Math.abs(this.y-v.y)}setLength(length){return this.normalize().multiplyScalar(length)}lerp(v,alpha){return this.x+=(v.x-this.x)*alpha,this.y+=(v.y-this.y)*alpha,this}lerpVectors(v1,v2,alpha){return this.x=v1.x+(v2.x-v1.x)*alpha,this.y=v1.y+(v2.y-v1.y)*alpha,this}equals(v){return v.x===this.x&&v.y===this.y}fromArray(array,offset=0){return this.x=array[offset],this.y=array[offset+1],this}toArray(array=[],offset=0){return array[offset]=this.x,array[offset+1]=this.y,array}fromBufferAttribute(attribute,index){return this.x=attribute.getX(index),this.y=attribute.getY(index),this}rotateAround(center,angle){const c=Math.cos(angle),s=Math.sin(angle),x=this.x-center.x,y=this.y-center.y;return this.x=x*c-y*s+center.x,this.y=x*s+y*c+center.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Matrix3{static{__name(this,"Matrix3")}constructor(n11,n12,n13,n21,n22,n23,n31,n32,n33){Matrix3.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],n11!==void 0&&this.set(n11,n12,n13,n21,n22,n23,n31,n32,n33)}set(n11,n12,n13,n21,n22,n23,n31,n32,n33){const te=this.elements;return te[0]=n11,te[1]=n21,te[2]=n31,te[3]=n12,te[4]=n22,te[5]=n32,te[6]=n13,te[7]=n23,te[8]=n33,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(m){const te=this.elements,me=m.elements;return te[0]=me[0],te[1]=me[1],te[2]=me[2],te[3]=me[3],te[4]=me[4],te[5]=me[5],te[6]=me[6],te[7]=me[7],te[8]=me[8],this}extractBasis(xAxis,yAxis,zAxis){return xAxis.setFromMatrix3Column(this,0),yAxis.setFromMatrix3Column(this,1),zAxis.setFromMatrix3Column(this,2),this}setFromMatrix4(m){const me=m.elements;return this.set(me[0],me[4],me[8],me[1],me[5],me[9],me[2],me[6],me[10]),this}multiply(m){return this.multiplyMatrices(this,m)}premultiply(m){return this.multiplyMatrices(m,this)}multiplyMatrices(a,b){const ae=a.elements,be=b.elements,te=this.elements,a11=ae[0],a12=ae[3],a13=ae[6],a21=ae[1],a22=ae[4],a23=ae[7],a31=ae[2],a32=ae[5],a33=ae[8],b11=be[0],b12=be[3],b13=be[6],b21=be[1],b22=be[4],b23=be[7],b31=be[2],b32=be[5],b33=be[8];return te[0]=a11*b11+a12*b21+a13*b31,te[3]=a11*b12+a12*b22+a13*b32,te[6]=a11*b13+a12*b23+a13*b33,te[1]=a21*b11+a22*b21+a23*b31,te[4]=a21*b12+a22*b22+a23*b32,te[7]=a21*b13+a22*b23+a23*b33,te[2]=a31*b11+a32*b21+a33*b31,te[5]=a31*b12+a32*b22+a33*b32,te[8]=a31*b13+a32*b23+a33*b33,this}multiplyScalar(s){const te=this.elements;return te[0]*=s,te[3]*=s,te[6]*=s,te[1]*=s,te[4]*=s,te[7]*=s,te[2]*=s,te[5]*=s,te[8]*=s,this}determinant(){const te=this.elements,a=te[0],b=te[1],c=te[2],d=te[3],e=te[4],f=te[5],g=te[6],h=te[7],i=te[8];return a*e*i-a*f*h-b*d*i+b*f*g+c*d*h-c*e*g}invert(){const te=this.elements,n11=te[0],n21=te[1],n31=te[2],n12=te[3],n22=te[4],n32=te[5],n13=te[6],n23=te[7],n33=te[8],t11=n33*n22-n32*n23,t12=n32*n13-n33*n12,t13=n23*n12-n22*n13,det=n11*t11+n21*t12+n31*t13;if(det===0)return this.set(0,0,0,0,0,0,0,0,0);const detInv=1/det;return te[0]=t11*detInv,te[1]=(n31*n23-n33*n21)*detInv,te[2]=(n32*n21-n31*n22)*detInv,te[3]=t12*detInv,te[4]=(n33*n11-n31*n13)*detInv,te[5]=(n31*n12-n32*n11)*detInv,te[6]=t13*detInv,te[7]=(n21*n13-n23*n11)*detInv,te[8]=(n22*n11-n21*n12)*detInv,this}transpose(){let tmp;const m=this.elements;return tmp=m[1],m[1]=m[3],m[3]=tmp,tmp=m[2],m[2]=m[6],m[6]=tmp,tmp=m[5],m[5]=m[7],m[7]=tmp,this}getNormalMatrix(matrix4){return this.setFromMatrix4(matrix4).invert().transpose()}transposeIntoArray(r){const m=this.elements;return r[0]=m[0],r[1]=m[3],r[2]=m[6],r[3]=m[1],r[4]=m[4],r[5]=m[7],r[6]=m[2],r[7]=m[5],r[8]=m[8],this}setUvTransform(tx,ty,sx,sy,rotation,cx,cy){const c=Math.cos(rotation),s=Math.sin(rotation);return this.set(sx*c,sx*s,-sx*(c*cx+s*cy)+cx+tx,-sy*s,sy*c,-sy*(-s*cx+c*cy)+cy+ty,0,0,1),this}scale(sx,sy){return this.premultiply(_m3.makeScale(sx,sy)),this}rotate(theta){return this.premultiply(_m3.makeRotation(-theta)),this}translate(tx,ty){return this.premultiply(_m3.makeTranslation(tx,ty)),this}makeTranslation(x,y){return x.isVector2?this.set(1,0,x.x,0,1,x.y,0,0,1):this.set(1,0,x,0,1,y,0,0,1),this}makeRotation(theta){const c=Math.cos(theta),s=Math.sin(theta);return this.set(c,-s,0,s,c,0,0,0,1),this}makeScale(x,y){return this.set(x,0,0,0,y,0,0,0,1),this}equals(matrix){const te=this.elements,me=matrix.elements;for(let i=0;i<9;i++)if(te[i]!==me[i])return!1;return!0}fromArray(array,offset=0){for(let i=0;i<9;i++)this.elements[i]=array[i+offset];return this}toArray(array=[],offset=0){const te=this.elements;return array[offset]=te[0],array[offset+1]=te[1],array[offset+2]=te[2],array[offset+3]=te[3],array[offset+4]=te[4],array[offset+5]=te[5],array[offset+6]=te[6],array[offset+7]=te[7],array[offset+8]=te[8],array}clone(){return new this.constructor().fromArray(this.elements)}}const _m3=new Matrix3;function arrayNeedsUint32(array){for(let i=array.length-1;i>=0;--i)if(array[i]>=65535)return!0;return!1}__name(arrayNeedsUint32,"arrayNeedsUint32");function createElementNS(name){return document.createElementNS("http://www.w3.org/1999/xhtml",name)}__name(createElementNS,"createElementNS");function createCanvasElement(){const canvas=createElementNS("canvas");return canvas.style.display="block",canvas}__name(createCanvasElement,"createCanvasElement");const _cache={};function warnOnce(message){message in _cache||(_cache[message]=!0,console.warn(message))}__name(warnOnce,"warnOnce");function probeAsync(gl,sync,interval){return new Promise(function(resolve,reject){function probe(){switch(gl.clientWaitSync(sync,gl.SYNC_FLUSH_COMMANDS_BIT,0)){case gl.WAIT_FAILED:reject();break;case gl.TIMEOUT_EXPIRED:setTimeout(probe,interval);break;default:resolve()}}__name(probe,"probe"),setTimeout(probe,interval)})}__name(probeAsync,"probeAsync");function toNormalizedProjectionMatrix(projectionMatrix){const m=projectionMatrix.elements;m[2]=.5*m[2]+.5*m[3],m[6]=.5*m[6]+.5*m[7],m[10]=.5*m[10]+.5*m[11],m[14]=.5*m[14]+.5*m[15]}__name(toNormalizedProjectionMatrix,"toNormalizedProjectionMatrix");function toReversedProjectionMatrix(projectionMatrix){const m=projectionMatrix.elements;m[11]===-1?(m[10]=-m[10]-1,m[14]=-m[14]):(m[10]=-m[10],m[14]=-m[14]+1)}__name(toReversedProjectionMatrix,"toReversedProjectionMatrix");const ColorManagement={enabled:!0,workingColorSpace:LinearSRGBColorSpace,spaces:{},convert:__name(function(color,sourceColorSpace,targetColorSpace){return this.enabled===!1||sourceColorSpace===targetColorSpace||!sourceColorSpace||!targetColorSpace||(this.spaces[sourceColorSpace].transfer===SRGBTransfer&&(color.r=SRGBToLinear(color.r),color.g=SRGBToLinear(color.g),color.b=SRGBToLinear(color.b)),this.spaces[sourceColorSpace].primaries!==this.spaces[targetColorSpace].primaries&&(color.applyMatrix3(this.spaces[sourceColorSpace].toXYZ),color.applyMatrix3(this.spaces[targetColorSpace].fromXYZ)),this.spaces[targetColorSpace].transfer===SRGBTransfer&&(color.r=LinearToSRGB(color.r),color.g=LinearToSRGB(color.g),color.b=LinearToSRGB(color.b))),color},"convert"),fromWorkingColorSpace:__name(function(color,targetColorSpace){return this.convert(color,this.workingColorSpace,targetColorSpace)},"fromWorkingColorSpace"),toWorkingColorSpace:__name(function(color,sourceColorSpace){return this.convert(color,sourceColorSpace,this.workingColorSpace)},"toWorkingColorSpace"),getPrimaries:__name(function(colorSpace){return this.spaces[colorSpace].primaries},"getPrimaries"),getTransfer:__name(function(colorSpace){return colorSpace===NoColorSpace?LinearTransfer:this.spaces[colorSpace].transfer},"getTransfer"),getLuminanceCoefficients:__name(function(target,colorSpace=this.workingColorSpace){return target.fromArray(this.spaces[colorSpace].luminanceCoefficients)},"getLuminanceCoefficients"),define:__name(function(colorSpaces){Object.assign(this.spaces,colorSpaces)},"define"),_getMatrix:__name(function(targetMatrix,sourceColorSpace,targetColorSpace){return targetMatrix.copy(this.spaces[sourceColorSpace].toXYZ).multiply(this.spaces[targetColorSpace].fromXYZ)},"_getMatrix"),_getDrawingBufferColorSpace:__name(function(colorSpace){return this.spaces[colorSpace].outputColorSpaceConfig.drawingBufferColorSpace},"_getDrawingBufferColorSpace"),_getUnpackColorSpace:__name(function(colorSpace=this.workingColorSpace){return this.spaces[colorSpace].workingColorSpaceConfig.unpackColorSpace},"_getUnpackColorSpace")};function SRGBToLinear(c){return c<.04045?c*.0773993808:Math.pow(c*.9478672986+.0521327014,2.4)}__name(SRGBToLinear,"SRGBToLinear");function LinearToSRGB(c){return c<.0031308?c*12.92:1.055*Math.pow(c,.41666)-.055}__name(LinearToSRGB,"LinearToSRGB");const REC709_PRIMARIES=[.64,.33,.3,.6,.15,.06],REC709_LUMINANCE_COEFFICIENTS=[.2126,.7152,.0722],D65=[.3127,.329],LINEAR_REC709_TO_XYZ=new Matrix3().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),XYZ_TO_LINEAR_REC709=new Matrix3().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);ColorManagement.define({[LinearSRGBColorSpace]:{primaries:REC709_PRIMARIES,whitePoint:D65,transfer:LinearTransfer,toXYZ:LINEAR_REC709_TO_XYZ,fromXYZ:XYZ_TO_LINEAR_REC709,luminanceCoefficients:REC709_LUMINANCE_COEFFICIENTS,workingColorSpaceConfig:{unpackColorSpace:SRGBColorSpace},outputColorSpaceConfig:{drawingBufferColorSpace:SRGBColorSpace}},[SRGBColorSpace]:{primaries:REC709_PRIMARIES,whitePoint:D65,transfer:SRGBTransfer,toXYZ:LINEAR_REC709_TO_XYZ,fromXYZ:XYZ_TO_LINEAR_REC709,luminanceCoefficients:REC709_LUMINANCE_COEFFICIENTS,outputColorSpaceConfig:{drawingBufferColorSpace:SRGBColorSpace}}});let _canvas;class ImageUtils{static{__name(this,"ImageUtils")}static getDataURL(image){if(/^data:/i.test(image.src)||typeof HTMLCanvasElement>"u")return image.src;let canvas;if(image instanceof HTMLCanvasElement)canvas=image;else{_canvas===void 0&&(_canvas=createElementNS("canvas")),_canvas.width=image.width,_canvas.height=image.height;const context=_canvas.getContext("2d");image instanceof ImageData?context.putImageData(image,0,0):context.drawImage(image,0,0,image.width,image.height),canvas=_canvas}return canvas.width>2048||canvas.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",image),canvas.toDataURL("image/jpeg",.6)):canvas.toDataURL("image/png")}static sRGBToLinear(image){if(typeof HTMLImageElement<"u"&&image instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&image instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&image instanceof ImageBitmap){const canvas=createElementNS("canvas");canvas.width=image.width,canvas.height=image.height;const context=canvas.getContext("2d");context.drawImage(image,0,0,image.width,image.height);const imageData=context.getImageData(0,0,image.width,image.height),data=imageData.data;for(let i=0;i<data.length;i++)data[i]=SRGBToLinear(data[i]/255)*255;return context.putImageData(imageData,0,0),canvas}else if(image.data){const data=image.data.slice(0);for(let i=0;i<data.length;i++)data instanceof Uint8Array||data instanceof Uint8ClampedArray?data[i]=Math.floor(SRGBToLinear(data[i]/255)*255):data[i]=SRGBToLinear(data[i]);return{data,width:image.width,height:image.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),image}}let _sourceId=0;class Source{static{__name(this,"Source")}constructor(data=null){this.isSource=!0,Object.defineProperty(this,"id",{value:_sourceId++}),this.uuid=generateUUID(),this.data=data,this.dataReady=!0,this.version=0}set needsUpdate(value){value===!0&&this.version++}toJSON(meta){const isRootObject=meta===void 0||typeof meta=="string";if(!isRootObject&&meta.images[this.uuid]!==void 0)return meta.images[this.uuid];const output={uuid:this.uuid,url:""},data=this.data;if(data!==null){let url;if(Array.isArray(data)){url=[];for(let i=0,l=data.length;i<l;i++)data[i].isDataTexture?url.push(serializeImage(data[i].image)):url.push(serializeImage(data[i]))}else url=serializeImage(data);output.url=url}return isRootObject||(meta.images[this.uuid]=output),output}}function serializeImage(image){return typeof HTMLImageElement<"u"&&image instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&image instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&image instanceof ImageBitmap?ImageUtils.getDataURL(image):image.data?{data:Array.from(image.data),width:image.width,height:image.height,type:image.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}__name(serializeImage,"serializeImage");let _textureId=0;class Texture extends EventDispatcher{static{__name(this,"Texture")}constructor(image=Texture.DEFAULT_IMAGE,mapping=Texture.DEFAULT_MAPPING,wrapS=ClampToEdgeWrapping,wrapT=ClampToEdgeWrapping,magFilter=LinearFilter,minFilter=LinearMipmapLinearFilter,format=RGBAFormat,type=UnsignedByteType,anisotropy=Texture.DEFAULT_ANISOTROPY,colorSpace=NoColorSpace){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:_textureId++}),this.uuid=generateUUID(),this.name="",this.source=new Source(image),this.mipmaps=[],this.mapping=mapping,this.channel=0,this.wrapS=wrapS,this.wrapT=wrapT,this.magFilter=magFilter,this.minFilter=minFilter,this.anisotropy=anisotropy,this.format=format,this.internalFormat=null,this.type=type,this.offset=new Vector2(0,0),this.repeat=new Vector2(1,1),this.center=new Vector2(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Matrix3,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=colorSpace,this.userData={},this.version=0,this.onUpdate=null,this.isRenderTargetTexture=!1,this.pmremVersion=0}get image(){return this.source.data}set image(value=null){this.source.data=value}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return new this.constructor().copy(this)}copy(source){return this.name=source.name,this.source=source.source,this.mipmaps=source.mipmaps.slice(0),this.mapping=source.mapping,this.channel=source.channel,this.wrapS=source.wrapS,this.wrapT=source.wrapT,this.magFilter=source.magFilter,this.minFilter=source.minFilter,this.anisotropy=source.anisotropy,this.format=source.format,this.internalFormat=source.internalFormat,this.type=source.type,this.offset.copy(source.offset),this.repeat.copy(source.repeat),this.center.copy(source.center),this.rotation=source.rotation,this.matrixAutoUpdate=source.matrixAutoUpdate,this.matrix.copy(source.matrix),this.generateMipmaps=source.generateMipmaps,this.premultiplyAlpha=source.premultiplyAlpha,this.flipY=source.flipY,this.unpackAlignment=source.unpackAlignment,this.colorSpace=source.colorSpace,this.userData=JSON.parse(JSON.stringify(source.userData)),this.needsUpdate=!0,this}toJSON(meta){const isRootObject=meta===void 0||typeof meta=="string";if(!isRootObject&&meta.textures[this.uuid]!==void 0)return meta.textures[this.uuid];const output={metadata:{version:4.6,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(meta).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(output.userData=this.userData),isRootObject||(meta.textures[this.uuid]=output),output}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(uv){if(this.mapping!==UVMapping)return uv;if(uv.applyMatrix3(this.matrix),uv.x<0||uv.x>1)switch(this.wrapS){case RepeatWrapping:uv.x=uv.x-Math.floor(uv.x);break;case ClampToEdgeWrapping:uv.x=uv.x<0?0:1;break;case MirroredRepeatWrapping:Math.abs(Math.floor(uv.x)%2)===1?uv.x=Math.ceil(uv.x)-uv.x:uv.x=uv.x-Math.floor(uv.x);break}if(uv.y<0||uv.y>1)switch(this.wrapT){case RepeatWrapping:uv.y=uv.y-Math.floor(uv.y);break;case ClampToEdgeWrapping:uv.y=uv.y<0?0:1;break;case MirroredRepeatWrapping:Math.abs(Math.floor(uv.y)%2)===1?uv.y=Math.ceil(uv.y)-uv.y:uv.y=uv.y-Math.floor(uv.y);break}return this.flipY&&(uv.y=1-uv.y),uv}set needsUpdate(value){value===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(value){value===!0&&this.pmremVersion++}}Texture.DEFAULT_IMAGE=null;Texture.DEFAULT_MAPPING=UVMapping;Texture.DEFAULT_ANISOTROPY=1;class Vector4{static{__name(this,"Vector4")}constructor(x=0,y=0,z=0,w=1){Vector4.prototype.isVector4=!0,this.x=x,this.y=y,this.z=z,this.w=w}get width(){return this.z}set width(value){this.z=value}get height(){return this.w}set height(value){this.w=value}set(x,y,z,w){return this.x=x,this.y=y,this.z=z,this.w=w,this}setScalar(scalar){return this.x=scalar,this.y=scalar,this.z=scalar,this.w=scalar,this}setX(x){return this.x=x,this}setY(y){return this.y=y,this}setZ(z){return this.z=z,this}setW(w){return this.w=w,this}setComponent(index,value){switch(index){case 0:this.x=value;break;case 1:this.y=value;break;case 2:this.z=value;break;case 3:this.w=value;break;default:throw new Error("index is out of range: "+index)}return this}getComponent(index){switch(index){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+index)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(v){return this.x=v.x,this.y=v.y,this.z=v.z,this.w=v.w!==void 0?v.w:1,this}add(v){return this.x+=v.x,this.y+=v.y,this.z+=v.z,this.w+=v.w,this}addScalar(s){return this.x+=s,this.y+=s,this.z+=s,this.w+=s,this}addVectors(a,b){return this.x=a.x+b.x,this.y=a.y+b.y,this.z=a.z+b.z,this.w=a.w+b.w,this}addScaledVector(v,s){return this.x+=v.x*s,this.y+=v.y*s,this.z+=v.z*s,this.w+=v.w*s,this}sub(v){return this.x-=v.x,this.y-=v.y,this.z-=v.z,this.w-=v.w,this}subScalar(s){return this.x-=s,this.y-=s,this.z-=s,this.w-=s,this}subVectors(a,b){return this.x=a.x-b.x,this.y=a.y-b.y,this.z=a.z-b.z,this.w=a.w-b.w,this}multiply(v){return this.x*=v.x,this.y*=v.y,this.z*=v.z,this.w*=v.w,this}multiplyScalar(scalar){return this.x*=scalar,this.y*=scalar,this.z*=scalar,this.w*=scalar,this}applyMatrix4(m){const x=this.x,y=this.y,z=this.z,w=this.w,e=m.elements;return this.x=e[0]*x+e[4]*y+e[8]*z+e[12]*w,this.y=e[1]*x+e[5]*y+e[9]*z+e[13]*w,this.z=e[2]*x+e[6]*y+e[10]*z+e[14]*w,this.w=e[3]*x+e[7]*y+e[11]*z+e[15]*w,this}divide(v){return this.x/=v.x,this.y/=v.y,this.z/=v.z,this.w/=v.w,this}divideScalar(scalar){return this.multiplyScalar(1/scalar)}setAxisAngleFromQuaternion(q){this.w=2*Math.acos(q.w);const s=Math.sqrt(1-q.w*q.w);return s<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=q.x/s,this.y=q.y/s,this.z=q.z/s),this}setAxisAngleFromRotationMatrix(m){let angle,x,y,z;const te=m.elements,m11=te[0],m12=te[4],m13=te[8],m21=te[1],m22=te[5],m23=te[9],m31=te[2],m32=te[6],m33=te[10];if(Math.abs(m12-m21)<.01&&Math.abs(m13-m31)<.01&&Math.abs(m23-m32)<.01){if(Math.abs(m12+m21)<.1&&Math.abs(m13+m31)<.1&&Math.abs(m23+m32)<.1&&Math.abs(m11+m22+m33-3)<.1)return this.set(1,0,0,0),this;angle=Math.PI;const xx=(m11+1)/2,yy=(m22+1)/2,zz=(m33+1)/2,xy=(m12+m21)/4,xz=(m13+m31)/4,yz=(m23+m32)/4;return xx>yy&&xx>zz?xx<.01?(x=0,y=.707106781,z=.707106781):(x=Math.sqrt(xx),y=xy/x,z=xz/x):yy>zz?yy<.01?(x=.707106781,y=0,z=.707106781):(y=Math.sqrt(yy),x=xy/y,z=yz/y):zz<.01?(x=.707106781,y=.707106781,z=0):(z=Math.sqrt(zz),x=xz/z,y=yz/z),this.set(x,y,z,angle),this}let s=Math.sqrt((m32-m23)*(m32-m23)+(m13-m31)*(m13-m31)+(m21-m12)*(m21-m12));return Math.abs(s)<.001&&(s=1),this.x=(m32-m23)/s,this.y=(m13-m31)/s,this.z=(m21-m12)/s,this.w=Math.acos((m11+m22+m33-1)/2),this}setFromMatrixPosition(m){const e=m.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this.w=e[15],this}min(v){return this.x=Math.min(this.x,v.x),this.y=Math.min(this.y,v.y),this.z=Math.min(this.z,v.z),this.w=Math.min(this.w,v.w),this}max(v){return this.x=Math.max(this.x,v.x),this.y=Math.max(this.y,v.y),this.z=Math.max(this.z,v.z),this.w=Math.max(this.w,v.w),this}clamp(min,max2){return this.x=Math.max(min.x,Math.min(max2.x,this.x)),this.y=Math.max(min.y,Math.min(max2.y,this.y)),this.z=Math.max(min.z,Math.min(max2.z,this.z)),this.w=Math.max(min.w,Math.min(max2.w,this.w)),this}clampScalar(minVal,maxVal){return this.x=Math.max(minVal,Math.min(maxVal,this.x)),this.y=Math.max(minVal,Math.min(maxVal,this.y)),this.z=Math.max(minVal,Math.min(maxVal,this.z)),this.w=Math.max(minVal,Math.min(maxVal,this.w)),this}clampLength(min,max2){const length=this.length();return this.divideScalar(length||1).multiplyScalar(Math.max(min,Math.min(max2,length)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(v){return this.x*v.x+this.y*v.y+this.z*v.z+this.w*v.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(length){return this.normalize().multiplyScalar(length)}lerp(v,alpha){return this.x+=(v.x-this.x)*alpha,this.y+=(v.y-this.y)*alpha,this.z+=(v.z-this.z)*alpha,this.w+=(v.w-this.w)*alpha,this}lerpVectors(v1,v2,alpha){return this.x=v1.x+(v2.x-v1.x)*alpha,this.y=v1.y+(v2.y-v1.y)*alpha,this.z=v1.z+(v2.z-v1.z)*alpha,this.w=v1.w+(v2.w-v1.w)*alpha,this}equals(v){return v.x===this.x&&v.y===this.y&&v.z===this.z&&v.w===this.w}fromArray(array,offset=0){return this.x=array[offset],this.y=array[offset+1],this.z=array[offset+2],this.w=array[offset+3],this}toArray(array=[],offset=0){return array[offset]=this.x,array[offset+1]=this.y,array[offset+2]=this.z,array[offset+3]=this.w,array}fromBufferAttribute(attribute,index){return this.x=attribute.getX(index),this.y=attribute.getY(index),this.z=attribute.getZ(index),this.w=attribute.getW(index),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class RenderTarget extends EventDispatcher{static{__name(this,"RenderTarget")}constructor(width=1,height=1,options={}){super(),this.isRenderTarget=!0,this.width=width,this.height=height,this.depth=1,this.scissor=new Vector4(0,0,width,height),this.scissorTest=!1,this.viewport=new Vector4(0,0,width,height);const image={width,height,depth:1};options=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:LinearFilter,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1},options);const texture=new Texture(image,options.mapping,options.wrapS,options.wrapT,options.magFilter,options.minFilter,options.format,options.type,options.anisotropy,options.colorSpace);texture.flipY=!1,texture.generateMipmaps=options.generateMipmaps,texture.internalFormat=options.internalFormat,this.textures=[];const count=options.count;for(let i=0;i<count;i++)this.textures[i]=texture.clone(),this.textures[i].isRenderTargetTexture=!0;this.depthBuffer=options.depthBuffer,this.stencilBuffer=options.stencilBuffer,this.resolveDepthBuffer=options.resolveDepthBuffer,this.resolveStencilBuffer=options.resolveStencilBuffer,this.depthTexture=options.depthTexture,this.samples=options.samples}get texture(){return this.textures[0]}set texture(value){this.textures[0]=value}setSize(width,height,depth=1){if(this.width!==width||this.height!==height||this.depth!==depth){this.width=width,this.height=height,this.depth=depth;for(let i=0,il=this.textures.length;i<il;i++)this.textures[i].image.width=width,this.textures[i].image.height=height,this.textures[i].image.depth=depth;this.dispose()}this.viewport.set(0,0,width,height),this.scissor.set(0,0,width,height)}clone(){return new this.constructor().copy(this)}copy(source){this.width=source.width,this.height=source.height,this.depth=source.depth,this.scissor.copy(source.scissor),this.scissorTest=source.scissorTest,this.viewport.copy(source.viewport),this.textures.length=0;for(let i=0,il=source.textures.length;i<il;i++)this.textures[i]=source.textures[i].clone(),this.textures[i].isRenderTargetTexture=!0;const image=Object.assign({},source.texture.image);return this.texture.source=new Source(image),this.depthBuffer=source.depthBuffer,this.stencilBuffer=source.stencilBuffer,this.resolveDepthBuffer=source.resolveDepthBuffer,this.resolveStencilBuffer=source.resolveStencilBuffer,source.depthTexture!==null&&(this.depthTexture=source.depthTexture.clone()),this.samples=source.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class WebGLRenderTarget extends RenderTarget{static{__name(this,"WebGLRenderTarget")}constructor(width=1,height=1,options={}){super(width,height,options),this.isWebGLRenderTarget=!0}}class DataArrayTexture extends Texture{static{__name(this,"DataArrayTexture")}constructor(data=null,width=1,height=1,depth=1){super(null),this.isDataArrayTexture=!0,this.image={data,width,height,depth},this.magFilter=NearestFilter,this.minFilter=NearestFilter,this.wrapR=ClampToEdgeWrapping,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(layerIndex){this.layerUpdates.add(layerIndex)}clearLayerUpdates(){this.layerUpdates.clear()}}class Data3DTexture extends Texture{static{__name(this,"Data3DTexture")}constructor(data=null,width=1,height=1,depth=1){super(null),this.isData3DTexture=!0,this.image={data,width,height,depth},this.magFilter=NearestFilter,this.minFilter=NearestFilter,this.wrapR=ClampToEdgeWrapping,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Quaternion{static{__name(this,"Quaternion")}constructor(x=0,y=0,z=0,w=1){this.isQuaternion=!0,this._x=x,this._y=y,this._z=z,this._w=w}static slerpFlat(dst,dstOffset,src0,srcOffset0,src1,srcOffset1,t2){let x0=src0[srcOffset0+0],y0=src0[srcOffset0+1],z0=src0[srcOffset0+2],w0=src0[srcOffset0+3];const x1=src1[srcOffset1+0],y1=src1[srcOffset1+1],z1=src1[srcOffset1+2],w1=src1[srcOffset1+3];if(t2===0){dst[dstOffset+0]=x0,dst[dstOffset+1]=y0,dst[dstOffset+2]=z0,dst[dstOffset+3]=w0;return}if(t2===1){dst[dstOffset+0]=x1,dst[dstOffset+1]=y1,dst[dstOffset+2]=z1,dst[dstOffset+3]=w1;return}if(w0!==w1||x0!==x1||y0!==y1||z0!==z1){let s=1-t2;const cos=x0*x1+y0*y1+z0*z1+w0*w1,dir=cos>=0?1:-1,sqrSin=1-cos*cos;if(sqrSin>Number.EPSILON){const sin=Math.sqrt(sqrSin),len=Math.atan2(sin,cos*dir);s=Math.sin(s*len)/sin,t2=Math.sin(t2*len)/sin}const tDir=t2*dir;if(x0=x0*s+x1*tDir,y0=y0*s+y1*tDir,z0=z0*s+z1*tDir,w0=w0*s+w1*tDir,s===1-t2){const f=1/Math.sqrt(x0*x0+y0*y0+z0*z0+w0*w0);x0*=f,y0*=f,z0*=f,w0*=f}}dst[dstOffset]=x0,dst[dstOffset+1]=y0,dst[dstOffset+2]=z0,dst[dstOffset+3]=w0}static multiplyQuaternionsFlat(dst,dstOffset,src0,srcOffset0,src1,srcOffset1){const x0=src0[srcOffset0],y0=src0[srcOffset0+1],z0=src0[srcOffset0+2],w0=src0[srcOffset0+3],x1=src1[srcOffset1],y1=src1[srcOffset1+1],z1=src1[srcOffset1+2],w1=src1[srcOffset1+3];return dst[dstOffset]=x0*w1+w0*x1+y0*z1-z0*y1,dst[dstOffset+1]=y0*w1+w0*y1+z0*x1-x0*z1,dst[dstOffset+2]=z0*w1+w0*z1+x0*y1-y0*x1,dst[dstOffset+3]=w0*w1-x0*x1-y0*y1-z0*z1,dst}get x(){return this._x}set x(value){this._x=value,this._onChangeCallback()}get y(){return this._y}set y(value){this._y=value,this._onChangeCallback()}get z(){return this._z}set z(value){this._z=value,this._onChangeCallback()}get w(){return this._w}set w(value){this._w=value,this._onChangeCallback()}set(x,y,z,w){return this._x=x,this._y=y,this._z=z,this._w=w,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(quaternion){return this._x=quaternion.x,this._y=quaternion.y,this._z=quaternion.z,this._w=quaternion.w,this._onChangeCallback(),this}setFromEuler(euler,update=!0){const x=euler._x,y=euler._y,z=euler._z,order=euler._order,cos=Math.cos,sin=Math.sin,c1=cos(x/2),c2=cos(y/2),c3=cos(z/2),s1=sin(x/2),s2=sin(y/2),s3=sin(z/2);switch(order){case"XYZ":this._x=s1*c2*c3+c1*s2*s3,this._y=c1*s2*c3-s1*c2*s3,this._z=c1*c2*s3+s1*s2*c3,this._w=c1*c2*c3-s1*s2*s3;break;case"YXZ":this._x=s1*c2*c3+c1*s2*s3,this._y=c1*s2*c3-s1*c2*s3,this._z=c1*c2*s3-s1*s2*c3,this._w=c1*c2*c3+s1*s2*s3;break;case"ZXY":this._x=s1*c2*c3-c1*s2*s3,this._y=c1*s2*c3+s1*c2*s3,this._z=c1*c2*s3+s1*s2*c3,this._w=c1*c2*c3-s1*s2*s3;break;case"ZYX":this._x=s1*c2*c3-c1*s2*s3,this._y=c1*s2*c3+s1*c2*s3,this._z=c1*c2*s3-s1*s2*c3,this._w=c1*c2*c3+s1*s2*s3;break;case"YZX":this._x=s1*c2*c3+c1*s2*s3,this._y=c1*s2*c3+s1*c2*s3,this._z=c1*c2*s3-s1*s2*c3,this._w=c1*c2*c3-s1*s2*s3;break;case"XZY":this._x=s1*c2*c3-c1*s2*s3,this._y=c1*s2*c3-s1*c2*s3,this._z=c1*c2*s3+s1*s2*c3,this._w=c1*c2*c3+s1*s2*s3;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+order)}return update===!0&&this._onChangeCallback(),this}setFromAxisAngle(axis,angle){const halfAngle=angle/2,s=Math.sin(halfAngle);return this._x=axis.x*s,this._y=axis.y*s,this._z=axis.z*s,this._w=Math.cos(halfAngle),this._onChangeCallback(),this}setFromRotationMatrix(m){const te=m.elements,m11=te[0],m12=te[4],m13=te[8],m21=te[1],m22=te[5],m23=te[9],m31=te[2],m32=te[6],m33=te[10],trace=m11+m22+m33;if(trace>0){const s=.5/Math.sqrt(trace+1);this._w=.25/s,this._x=(m32-m23)*s,this._y=(m13-m31)*s,this._z=(m21-m12)*s}else if(m11>m22&&m11>m33){const s=2*Math.sqrt(1+m11-m22-m33);this._w=(m32-m23)/s,this._x=.25*s,this._y=(m12+m21)/s,this._z=(m13+m31)/s}else if(m22>m33){const s=2*Math.sqrt(1+m22-m11-m33);this._w=(m13-m31)/s,this._x=(m12+m21)/s,this._y=.25*s,this._z=(m23+m32)/s}else{const s=2*Math.sqrt(1+m33-m11-m22);this._w=(m21-m12)/s,this._x=(m13+m31)/s,this._y=(m23+m32)/s,this._z=.25*s}return this._onChangeCallback(),this}setFromUnitVectors(vFrom,vTo){let r=vFrom.dot(vTo)+1;return r<Number.EPSILON?(r=0,Math.abs(vFrom.x)>Math.abs(vFrom.z)?(this._x=-vFrom.y,this._y=vFrom.x,this._z=0,this._w=r):(this._x=0,this._y=-vFrom.z,this._z=vFrom.y,this._w=r)):(this._x=vFrom.y*vTo.z-vFrom.z*vTo.y,this._y=vFrom.z*vTo.x-vFrom.x*vTo.z,this._z=vFrom.x*vTo.y-vFrom.y*vTo.x,this._w=r),this.normalize()}angleTo(q){return 2*Math.acos(Math.abs(clamp(this.dot(q),-1,1)))}rotateTowards(q,step){const angle=this.angleTo(q);if(angle===0)return this;const t2=Math.min(1,step/angle);return this.slerp(q,t2),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(v){return this._x*v._x+this._y*v._y+this._z*v._z+this._w*v._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let l=this.length();return l===0?(this._x=0,this._y=0,this._z=0,this._w=1):(l=1/l,this._x=this._x*l,this._y=this._y*l,this._z=this._z*l,this._w=this._w*l),this._onChangeCallback(),this}multiply(q){return this.multiplyQuaternions(this,q)}premultiply(q){return this.multiplyQuaternions(q,this)}multiplyQuaternions(a,b){const qax=a._x,qay=a._y,qaz=a._z,qaw=a._w,qbx=b._x,qby=b._y,qbz=b._z,qbw=b._w;return this._x=qax*qbw+qaw*qbx+qay*qbz-qaz*qby,this._y=qay*qbw+qaw*qby+qaz*qbx-qax*qbz,this._z=qaz*qbw+qaw*qbz+qax*qby-qay*qbx,this._w=qaw*qbw-qax*qbx-qay*qby-qaz*qbz,this._onChangeCallback(),this}slerp(qb,t2){if(t2===0)return this;if(t2===1)return this.copy(qb);const x=this._x,y=this._y,z=this._z,w=this._w;let cosHalfTheta=w*qb._w+x*qb._x+y*qb._y+z*qb._z;if(cosHalfTheta<0?(this._w=-qb._w,this._x=-qb._x,this._y=-qb._y,this._z=-qb._z,cosHalfTheta=-cosHalfTheta):this.copy(qb),cosHalfTheta>=1)return this._w=w,this._x=x,this._y=y,this._z=z,this;const sqrSinHalfTheta=1-cosHalfTheta*cosHalfTheta;if(sqrSinHalfTheta<=Number.EPSILON){const s=1-t2;return this._w=s*w+t2*this._w,this._x=s*x+t2*this._x,this._y=s*y+t2*this._y,this._z=s*z+t2*this._z,this.normalize(),this}const sinHalfTheta=Math.sqrt(sqrSinHalfTheta),halfTheta=Math.atan2(sinHalfTheta,cosHalfTheta),ratioA=Math.sin((1-t2)*halfTheta)/sinHalfTheta,ratioB=Math.sin(t2*halfTheta)/sinHalfTheta;return this._w=w*ratioA+this._w*ratioB,this._x=x*ratioA+this._x*ratioB,this._y=y*ratioA+this._y*ratioB,this._z=z*ratioA+this._z*ratioB,this._onChangeCallback(),this}slerpQuaternions(qa,qb,t2){return this.copy(qa).slerp(qb,t2)}random(){const theta1=2*Math.PI*Math.random(),theta2=2*Math.PI*Math.random(),x0=Math.random(),r1=Math.sqrt(1-x0),r2=Math.sqrt(x0);return this.set(r1*Math.sin(theta1),r1*Math.cos(theta1),r2*Math.sin(theta2),r2*Math.cos(theta2))}equals(quaternion){return quaternion._x===this._x&&quaternion._y===this._y&&quaternion._z===this._z&&quaternion._w===this._w}fromArray(array,offset=0){return this._x=array[offset],this._y=array[offset+1],this._z=array[offset+2],this._w=array[offset+3],this._onChangeCallback(),this}toArray(array=[],offset=0){return array[offset]=this._x,array[offset+1]=this._y,array[offset+2]=this._z,array[offset+3]=this._w,array}fromBufferAttribute(attribute,index){return this._x=attribute.getX(index),this._y=attribute.getY(index),this._z=attribute.getZ(index),this._w=attribute.getW(index),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(callback){return this._onChangeCallback=callback,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class Vector3{static{__name(this,"Vector3")}constructor(x=0,y=0,z=0){Vector3.prototype.isVector3=!0,this.x=x,this.y=y,this.z=z}set(x,y,z){return z===void 0&&(z=this.z),this.x=x,this.y=y,this.z=z,this}setScalar(scalar){return this.x=scalar,this.y=scalar,this.z=scalar,this}setX(x){return this.x=x,this}setY(y){return this.y=y,this}setZ(z){return this.z=z,this}setComponent(index,value){switch(index){case 0:this.x=value;break;case 1:this.y=value;break;case 2:this.z=value;break;default:throw new Error("index is out of range: "+index)}return this}getComponent(index){switch(index){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+index)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(v){return this.x=v.x,this.y=v.y,this.z=v.z,this}add(v){return this.x+=v.x,this.y+=v.y,this.z+=v.z,this}addScalar(s){return this.x+=s,this.y+=s,this.z+=s,this}addVectors(a,b){return this.x=a.x+b.x,this.y=a.y+b.y,this.z=a.z+b.z,this}addScaledVector(v,s){return this.x+=v.x*s,this.y+=v.y*s,this.z+=v.z*s,this}sub(v){return this.x-=v.x,this.y-=v.y,this.z-=v.z,this}subScalar(s){return this.x-=s,this.y-=s,this.z-=s,this}subVectors(a,b){return this.x=a.x-b.x,this.y=a.y-b.y,this.z=a.z-b.z,this}multiply(v){return this.x*=v.x,this.y*=v.y,this.z*=v.z,this}multiplyScalar(scalar){return this.x*=scalar,this.y*=scalar,this.z*=scalar,this}multiplyVectors(a,b){return this.x=a.x*b.x,this.y=a.y*b.y,this.z=a.z*b.z,this}applyEuler(euler){return this.applyQuaternion(_quaternion$4.setFromEuler(euler))}applyAxisAngle(axis,angle){return this.applyQuaternion(_quaternion$4.setFromAxisAngle(axis,angle))}applyMatrix3(m){const x=this.x,y=this.y,z=this.z,e=m.elements;return this.x=e[0]*x+e[3]*y+e[6]*z,this.y=e[1]*x+e[4]*y+e[7]*z,this.z=e[2]*x+e[5]*y+e[8]*z,this}applyNormalMatrix(m){return this.applyMatrix3(m).normalize()}applyMatrix4(m){const x=this.x,y=this.y,z=this.z,e=m.elements,w=1/(e[3]*x+e[7]*y+e[11]*z+e[15]);return this.x=(e[0]*x+e[4]*y+e[8]*z+e[12])*w,this.y=(e[1]*x+e[5]*y+e[9]*z+e[13])*w,this.z=(e[2]*x+e[6]*y+e[10]*z+e[14])*w,this}applyQuaternion(q){const vx=this.x,vy=this.y,vz=this.z,qx=q.x,qy=q.y,qz=q.z,qw=q.w,tx=2*(qy*vz-qz*vy),ty=2*(qz*vx-qx*vz),tz=2*(qx*vy-qy*vx);return this.x=vx+qw*tx+qy*tz-qz*ty,this.y=vy+qw*ty+qz*tx-qx*tz,this.z=vz+qw*tz+qx*ty-qy*tx,this}project(camera){return this.applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix)}unproject(camera){return this.applyMatrix4(camera.projectionMatrixInverse).applyMatrix4(camera.matrixWorld)}transformDirection(m){const x=this.x,y=this.y,z=this.z,e=m.elements;return this.x=e[0]*x+e[4]*y+e[8]*z,this.y=e[1]*x+e[5]*y+e[9]*z,this.z=e[2]*x+e[6]*y+e[10]*z,this.normalize()}divide(v){return this.x/=v.x,this.y/=v.y,this.z/=v.z,this}divideScalar(scalar){return this.multiplyScalar(1/scalar)}min(v){return this.x=Math.min(this.x,v.x),this.y=Math.min(this.y,v.y),this.z=Math.min(this.z,v.z),this}max(v){return this.x=Math.max(this.x,v.x),this.y=Math.max(this.y,v.y),this.z=Math.max(this.z,v.z),this}clamp(min,max2){return this.x=Math.max(min.x,Math.min(max2.x,this.x)),this.y=Math.max(min.y,Math.min(max2.y,this.y)),this.z=Math.max(min.z,Math.min(max2.z,this.z)),this}clampScalar(minVal,maxVal){return this.x=Math.max(minVal,Math.min(maxVal,this.x)),this.y=Math.max(minVal,Math.min(maxVal,this.y)),this.z=Math.max(minVal,Math.min(maxVal,this.z)),this}clampLength(min,max2){const length=this.length();return this.divideScalar(length||1).multiplyScalar(Math.max(min,Math.min(max2,length)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(v){return this.x*v.x+this.y*v.y+this.z*v.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(length){return this.normalize().multiplyScalar(length)}lerp(v,alpha){return this.x+=(v.x-this.x)*alpha,this.y+=(v.y-this.y)*alpha,this.z+=(v.z-this.z)*alpha,this}lerpVectors(v1,v2,alpha){return this.x=v1.x+(v2.x-v1.x)*alpha,this.y=v1.y+(v2.y-v1.y)*alpha,this.z=v1.z+(v2.z-v1.z)*alpha,this}cross(v){return this.crossVectors(this,v)}crossVectors(a,b){const ax=a.x,ay=a.y,az=a.z,bx=b.x,by=b.y,bz=b.z;return this.x=ay*bz-az*by,this.y=az*bx-ax*bz,this.z=ax*by-ay*bx,this}projectOnVector(v){const denominator=v.lengthSq();if(denominator===0)return this.set(0,0,0);const scalar=v.dot(this)/denominator;return this.copy(v).multiplyScalar(scalar)}projectOnPlane(planeNormal){return _vector$c.copy(this).projectOnVector(planeNormal),this.sub(_vector$c)}reflect(normal){return this.sub(_vector$c.copy(normal).multiplyScalar(2*this.dot(normal)))}angleTo(v){const denominator=Math.sqrt(this.lengthSq()*v.lengthSq());if(denominator===0)return Math.PI/2;const theta=this.dot(v)/denominator;return Math.acos(clamp(theta,-1,1))}distanceTo(v){return Math.sqrt(this.distanceToSquared(v))}distanceToSquared(v){const dx=this.x-v.x,dy=this.y-v.y,dz=this.z-v.z;return dx*dx+dy*dy+dz*dz}manhattanDistanceTo(v){return Math.abs(this.x-v.x)+Math.abs(this.y-v.y)+Math.abs(this.z-v.z)}setFromSpherical(s){return this.setFromSphericalCoords(s.radius,s.phi,s.theta)}setFromSphericalCoords(radius,phi,theta){const sinPhiRadius=Math.sin(phi)*radius;return this.x=sinPhiRadius*Math.sin(theta),this.y=Math.cos(phi)*radius,this.z=sinPhiRadius*Math.cos(theta),this}setFromCylindrical(c){return this.setFromCylindricalCoords(c.radius,c.theta,c.y)}setFromCylindricalCoords(radius,theta,y){return this.x=radius*Math.sin(theta),this.y=y,this.z=radius*Math.cos(theta),this}setFromMatrixPosition(m){const e=m.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(m){const sx=this.setFromMatrixColumn(m,0).length(),sy=this.setFromMatrixColumn(m,1).length(),sz=this.setFromMatrixColumn(m,2).length();return this.x=sx,this.y=sy,this.z=sz,this}setFromMatrixColumn(m,index){return this.fromArray(m.elements,index*4)}setFromMatrix3Column(m,index){return this.fromArray(m.elements,index*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(c){return this.x=c.r,this.y=c.g,this.z=c.b,this}equals(v){return v.x===this.x&&v.y===this.y&&v.z===this.z}fromArray(array,offset=0){return this.x=array[offset],this.y=array[offset+1],this.z=array[offset+2],this}toArray(array=[],offset=0){return array[offset]=this.x,array[offset+1]=this.y,array[offset+2]=this.z,array}fromBufferAttribute(attribute,index){return this.x=attribute.getX(index),this.y=attribute.getY(index),this.z=attribute.getZ(index),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const theta=Math.random()*Math.PI*2,u=Math.random()*2-1,c=Math.sqrt(1-u*u);return this.x=c*Math.cos(theta),this.y=u,this.z=c*Math.sin(theta),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const _vector$c=new Vector3,_quaternion$4=new Quaternion;class Box3{static{__name(this,"Box3")}constructor(min=new Vector3(1/0,1/0,1/0),max2=new Vector3(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=min,this.max=max2}set(min,max2){return this.min.copy(min),this.max.copy(max2),this}setFromArray(array){this.makeEmpty();for(let i=0,il=array.length;i<il;i+=3)this.expandByPoint(_vector$b.fromArray(array,i));return this}setFromBufferAttribute(attribute){this.makeEmpty();for(let i=0,il=attribute.count;i<il;i++)this.expandByPoint(_vector$b.fromBufferAttribute(attribute,i));return this}setFromPoints(points){this.makeEmpty();for(let i=0,il=points.length;i<il;i++)this.expandByPoint(points[i]);return this}setFromCenterAndSize(center,size){const halfSize=_vector$b.copy(size).multiplyScalar(.5);return this.min.copy(center).sub(halfSize),this.max.copy(center).add(halfSize),this}setFromObject(object,precise=!1){return this.makeEmpty(),this.expandByObject(object,precise)}clone(){return new this.constructor().copy(this)}copy(box){return this.min.copy(box.min),this.max.copy(box.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(target){return this.isEmpty()?target.set(0,0,0):target.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(target){return this.isEmpty()?target.set(0,0,0):target.subVectors(this.max,this.min)}expandByPoint(point){return this.min.min(point),this.max.max(point),this}expandByVector(vector){return this.min.sub(vector),this.max.add(vector),this}expandByScalar(scalar){return this.min.addScalar(-scalar),this.max.addScalar(scalar),this}expandByObject(object,precise=!1){object.updateWorldMatrix(!1,!1);const geometry=object.geometry;if(geometry!==void 0){const positionAttribute=geometry.getAttribute("position");if(precise===!0&&positionAttribute!==void 0&&object.isInstancedMesh!==!0)for(let i=0,l=positionAttribute.count;i<l;i++)object.isMesh===!0?object.getVertexPosition(i,_vector$b):_vector$b.fromBufferAttribute(positionAttribute,i),_vector$b.applyMatrix4(object.matrixWorld),this.expandByPoint(_vector$b);else object.boundingBox!==void 0?(object.boundingBox===null&&object.computeBoundingBox(),_box$4.copy(object.boundingBox)):(geometry.boundingBox===null&&geometry.computeBoundingBox(),_box$4.copy(geometry.boundingBox)),_box$4.applyMatrix4(object.matrixWorld),this.union(_box$4)}const children=object.children;for(let i=0,l=children.length;i<l;i++)this.expandByObject(children[i],precise);return this}containsPoint(point){return point.x>=this.min.x&&point.x<=this.max.x&&point.y>=this.min.y&&point.y<=this.max.y&&point.z>=this.min.z&&point.z<=this.max.z}containsBox(box){return this.min.x<=box.min.x&&box.max.x<=this.max.x&&this.min.y<=box.min.y&&box.max.y<=this.max.y&&this.min.z<=box.min.z&&box.max.z<=this.max.z}getParameter(point,target){return target.set((point.x-this.min.x)/(this.max.x-this.min.x),(point.y-this.min.y)/(this.max.y-this.min.y),(point.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(box){return box.max.x>=this.min.x&&box.min.x<=this.max.x&&box.max.y>=this.min.y&&box.min.y<=this.max.y&&box.max.z>=this.min.z&&box.min.z<=this.max.z}intersectsSphere(sphere){return this.clampPoint(sphere.center,_vector$b),_vector$b.distanceToSquared(sphere.center)<=sphere.radius*sphere.radius}intersectsPlane(plane){let min,max2;return plane.normal.x>0?(min=plane.normal.x*this.min.x,max2=plane.normal.x*this.max.x):(min=plane.normal.x*this.max.x,max2=plane.normal.x*this.min.x),plane.normal.y>0?(min+=plane.normal.y*this.min.y,max2+=plane.normal.y*this.max.y):(min+=plane.normal.y*this.max.y,max2+=plane.normal.y*this.min.y),plane.normal.z>0?(min+=plane.normal.z*this.min.z,max2+=plane.normal.z*this.max.z):(min+=plane.normal.z*this.max.z,max2+=plane.normal.z*this.min.z),min<=-plane.constant&&max2>=-plane.constant}intersectsTriangle(triangle){if(this.isEmpty())return!1;this.getCenter(_center),_extents.subVectors(this.max,_center),_v0$3.subVectors(triangle.a,_center),_v1$7.subVectors(triangle.b,_center),_v2$4.subVectors(triangle.c,_center),_f0.subVectors(_v1$7,_v0$3),_f1.subVectors(_v2$4,_v1$7),_f2.subVectors(_v0$3,_v2$4);let axes=[0,-_f0.z,_f0.y,0,-_f1.z,_f1.y,0,-_f2.z,_f2.y,_f0.z,0,-_f0.x,_f1.z,0,-_f1.x,_f2.z,0,-_f2.x,-_f0.y,_f0.x,0,-_f1.y,_f1.x,0,-_f2.y,_f2.x,0];return!satForAxes(axes,_v0$3,_v1$7,_v2$4,_extents)||(axes=[1,0,0,0,1,0,0,0,1],!satForAxes(axes,_v0$3,_v1$7,_v2$4,_extents))?!1:(_triangleNormal.crossVectors(_f0,_f1),axes=[_triangleNormal.x,_triangleNormal.y,_triangleNormal.z],satForAxes(axes,_v0$3,_v1$7,_v2$4,_extents))}clampPoint(point,target){return target.copy(point).clamp(this.min,this.max)}distanceToPoint(point){return this.clampPoint(point,_vector$b).distanceTo(point)}getBoundingSphere(target){return this.isEmpty()?target.makeEmpty():(this.getCenter(target.center),target.radius=this.getSize(_vector$b).length()*.5),target}intersect(box){return this.min.max(box.min),this.max.min(box.max),this.isEmpty()&&this.makeEmpty(),this}union(box){return this.min.min(box.min),this.max.max(box.max),this}applyMatrix4(matrix){return this.isEmpty()?this:(_points[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(matrix),_points[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(matrix),_points[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(matrix),_points[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(matrix),_points[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(matrix),_points[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(matrix),_points[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(matrix),_points[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(matrix),this.setFromPoints(_points),this)}translate(offset){return this.min.add(offset),this.max.add(offset),this}equals(box){return box.min.equals(this.min)&&box.max.equals(this.max)}}const _points=[new Vector3,new Vector3,new Vector3,new Vector3,new Vector3,new Vector3,new Vector3,new Vector3],_vector$b=new Vector3,_box$4=new Box3,_v0$3=new Vector3,_v1$7=new Vector3,_v2$4=new Vector3,_f0=new Vector3,_f1=new Vector3,_f2=new Vector3,_center=new Vector3,_extents=new Vector3,_triangleNormal=new Vector3,_testAxis=new Vector3;function satForAxes(axes,v0,v1,v2,extents){for(let i=0,j=axes.length-3;i<=j;i+=3){_testAxis.fromArray(axes,i);const r=extents.x*Math.abs(_testAxis.x)+extents.y*Math.abs(_testAxis.y)+extents.z*Math.abs(_testAxis.z),p0=v0.dot(_testAxis),p1=v1.dot(_testAxis),p2=v2.dot(_testAxis);if(Math.max(-Math.max(p0,p1,p2),Math.min(p0,p1,p2))>r)return!1}return!0}__name(satForAxes,"satForAxes");const _box$3=new Box3,_v1$6=new Vector3,_v2$3=new Vector3;class Sphere{static{__name(this,"Sphere")}constructor(center=new Vector3,radius=-1){this.isSphere=!0,this.center=center,this.radius=radius}set(center,radius){return this.center.copy(center),this.radius=radius,this}setFromPoints(points,optionalCenter){const center=this.center;optionalCenter!==void 0?center.copy(optionalCenter):_box$3.setFromPoints(points).getCenter(center);let maxRadiusSq=0;for(let i=0,il=points.length;i<il;i++)maxRadiusSq=Math.max(maxRadiusSq,center.distanceToSquared(points[i]));return this.radius=Math.sqrt(maxRadiusSq),this}copy(sphere){return this.center.copy(sphere.center),this.radius=sphere.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(point){return point.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(point){return point.distanceTo(this.center)-this.radius}intersectsSphere(sphere){const radiusSum=this.radius+sphere.radius;return sphere.center.distanceToSquared(this.center)<=radiusSum*radiusSum}intersectsBox(box){return box.intersectsSphere(this)}intersectsPlane(plane){return Math.abs(plane.distanceToPoint(this.center))<=this.radius}clampPoint(point,target){const deltaLengthSq=this.center.distanceToSquared(point);return target.copy(point),deltaLengthSq>this.radius*this.radius&&(target.sub(this.center).normalize(),target.multiplyScalar(this.radius).add(this.center)),target}getBoundingBox(target){return this.isEmpty()?(target.makeEmpty(),target):(target.set(this.center,this.center),target.expandByScalar(this.radius),target)}applyMatrix4(matrix){return this.center.applyMatrix4(matrix),this.radius=this.radius*matrix.getMaxScaleOnAxis(),this}translate(offset){return this.center.add(offset),this}expandByPoint(point){if(this.isEmpty())return this.center.copy(point),this.radius=0,this;_v1$6.subVectors(point,this.center);const lengthSq=_v1$6.lengthSq();if(lengthSq>this.radius*this.radius){const length=Math.sqrt(lengthSq),delta=(length-this.radius)*.5;this.center.addScaledVector(_v1$6,delta/length),this.radius+=delta}return this}union(sphere){return sphere.isEmpty()?this:this.isEmpty()?(this.copy(sphere),this):(this.center.equals(sphere.center)===!0?this.radius=Math.max(this.radius,sphere.radius):(_v2$3.subVectors(sphere.center,this.center).setLength(sphere.radius),this.expandByPoint(_v1$6.copy(sphere.center).add(_v2$3)),this.expandByPoint(_v1$6.copy(sphere.center).sub(_v2$3))),this)}equals(sphere){return sphere.center.equals(this.center)&&sphere.radius===this.radius}clone(){return new this.constructor().copy(this)}}const _vector$a=new Vector3,_segCenter=new Vector3,_segDir=new Vector3,_diff=new Vector3,_edge1=new Vector3,_edge2=new Vector3,_normal$1=new Vector3;class Ray{static{__name(this,"Ray")}constructor(origin=new Vector3,direction=new Vector3(0,0,-1)){this.origin=origin,this.direction=direction}set(origin,direction){return this.origin.copy(origin),this.direction.copy(direction),this}copy(ray){return this.origin.copy(ray.origin),this.direction.copy(ray.direction),this}at(t2,target){return target.copy(this.origin).addScaledVector(this.direction,t2)}lookAt(v){return this.direction.copy(v).sub(this.origin).normalize(),this}recast(t2){return this.origin.copy(this.at(t2,_vector$a)),this}closestPointToPoint(point,target){target.subVectors(point,this.origin);const directionDistance=target.dot(this.direction);return directionDistance<0?target.copy(this.origin):target.copy(this.origin).addScaledVector(this.direction,directionDistance)}distanceToPoint(point){return Math.sqrt(this.distanceSqToPoint(point))}distanceSqToPoint(point){const directionDistance=_vector$a.subVectors(point,this.origin).dot(this.direction);return directionDistance<0?this.origin.distanceToSquared(point):(_vector$a.copy(this.origin).addScaledVector(this.direction,directionDistance),_vector$a.distanceToSquared(point))}distanceSqToSegment(v0,v1,optionalPointOnRay,optionalPointOnSegment){_segCenter.copy(v0).add(v1).multiplyScalar(.5),_segDir.copy(v1).sub(v0).normalize(),_diff.copy(this.origin).sub(_segCenter);const segExtent=v0.distanceTo(v1)*.5,a01=-this.direction.dot(_segDir),b0=_diff.dot(this.direction),b1=-_diff.dot(_segDir),c=_diff.lengthSq(),det=Math.abs(1-a01*a01);let s0,s1,sqrDist,extDet;if(det>0)if(s0=a01*b1-b0,s1=a01*b0-b1,extDet=segExtent*det,s0>=0)if(s1>=-extDet)if(s1<=extDet){const invDet=1/det;s0*=invDet,s1*=invDet,sqrDist=s0*(s0+a01*s1+2*b0)+s1*(a01*s0+s1+2*b1)+c}else s1=segExtent,s0=Math.max(0,-(a01*s1+b0)),sqrDist=-s0*s0+s1*(s1+2*b1)+c;else s1=-segExtent,s0=Math.max(0,-(a01*s1+b0)),sqrDist=-s0*s0+s1*(s1+2*b1)+c;else s1<=-extDet?(s0=Math.max(0,-(-a01*segExtent+b0)),s1=s0>0?-segExtent:Math.min(Math.max(-segExtent,-b1),segExtent),sqrDist=-s0*s0+s1*(s1+2*b1)+c):s1<=extDet?(s0=0,s1=Math.min(Math.max(-segExtent,-b1),segExtent),sqrDist=s1*(s1+2*b1)+c):(s0=Math.max(0,-(a01*segExtent+b0)),s1=s0>0?segExtent:Math.min(Math.max(-segExtent,-b1),segExtent),sqrDist=-s0*s0+s1*(s1+2*b1)+c);else s1=a01>0?-segExtent:segExtent,s0=Math.max(0,-(a01*s1+b0)),sqrDist=-s0*s0+s1*(s1+2*b1)+c;return optionalPointOnRay&&optionalPointOnRay.copy(this.origin).addScaledVector(this.direction,s0),optionalPointOnSegment&&optionalPointOnSegment.copy(_segCenter).addScaledVector(_segDir,s1),sqrDist}intersectSphere(sphere,target){_vector$a.subVectors(sphere.center,this.origin);const tca=_vector$a.dot(this.direction),d2=_vector$a.dot(_vector$a)-tca*tca,radius2=sphere.radius*sphere.radius;if(d2>radius2)return null;const thc=Math.sqrt(radius2-d2),t0=tca-thc,t1=tca+thc;return t1<0?null:t0<0?this.at(t1,target):this.at(t0,target)}intersectsSphere(sphere){return this.distanceSqToPoint(sphere.center)<=sphere.radius*sphere.radius}distanceToPlane(plane){const denominator=plane.normal.dot(this.direction);if(denominator===0)return plane.distanceToPoint(this.origin)===0?0:null;const t2=-(this.origin.dot(plane.normal)+plane.constant)/denominator;return t2>=0?t2:null}intersectPlane(plane,target){const t2=this.distanceToPlane(plane);return t2===null?null:this.at(t2,target)}intersectsPlane(plane){const distToPoint=plane.distanceToPoint(this.origin);return distToPoint===0||plane.normal.dot(this.direction)*distToPoint<0}intersectBox(box,target){let tmin,tmax,tymin,tymax,tzmin,tzmax;const invdirx=1/this.direction.x,invdiry=1/this.direction.y,invdirz=1/this.direction.z,origin=this.origin;return invdirx>=0?(tmin=(box.min.x-origin.x)*invdirx,tmax=(box.max.x-origin.x)*invdirx):(tmin=(box.max.x-origin.x)*invdirx,tmax=(box.min.x-origin.x)*invdirx),invdiry>=0?(tymin=(box.min.y-origin.y)*invdiry,tymax=(box.max.y-origin.y)*invdiry):(tymin=(box.max.y-origin.y)*invdiry,tymax=(box.min.y-origin.y)*invdiry),tmin>tymax||tymin>tmax||((tymin>tmin||isNaN(tmin))&&(tmin=tymin),(tymax<tmax||isNaN(tmax))&&(tmax=tymax),invdirz>=0?(tzmin=(box.min.z-origin.z)*invdirz,tzmax=(box.max.z-origin.z)*invdirz):(tzmin=(box.max.z-origin.z)*invdirz,tzmax=(box.min.z-origin.z)*invdirz),tmin>tzmax||tzmin>tmax)||((tzmin>tmin||tmin!==tmin)&&(tmin=tzmin),(tzmax<tmax||tmax!==tmax)&&(tmax=tzmax),tmax<0)?null:this.at(tmin>=0?tmin:tmax,target)}intersectsBox(box){return this.intersectBox(box,_vector$a)!==null}intersectTriangle(a,b,c,backfaceCulling,target){_edge1.subVectors(b,a),_edge2.subVectors(c,a),_normal$1.crossVectors(_edge1,_edge2);let DdN=this.direction.dot(_normal$1),sign2;if(DdN>0){if(backfaceCulling)return null;sign2=1}else if(DdN<0)sign2=-1,DdN=-DdN;else return null;_diff.subVectors(this.origin,a);const DdQxE2=sign2*this.direction.dot(_edge2.crossVectors(_diff,_edge2));if(DdQxE2<0)return null;const DdE1xQ=sign2*this.direction.dot(_edge1.cross(_diff));if(DdE1xQ<0||DdQxE2+DdE1xQ>DdN)return null;const QdN=-sign2*_diff.dot(_normal$1);return QdN<0?null:this.at(QdN/DdN,target)}applyMatrix4(matrix4){return this.origin.applyMatrix4(matrix4),this.direction.transformDirection(matrix4),this}equals(ray){return ray.origin.equals(this.origin)&&ray.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class Matrix4{static{__name(this,"Matrix4")}constructor(n11,n12,n13,n14,n21,n22,n23,n24,n31,n32,n33,n34,n41,n42,n43,n44){Matrix4.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],n11!==void 0&&this.set(n11,n12,n13,n14,n21,n22,n23,n24,n31,n32,n33,n34,n41,n42,n43,n44)}set(n11,n12,n13,n14,n21,n22,n23,n24,n31,n32,n33,n34,n41,n42,n43,n44){const te=this.elements;return te[0]=n11,te[4]=n12,te[8]=n13,te[12]=n14,te[1]=n21,te[5]=n22,te[9]=n23,te[13]=n24,te[2]=n31,te[6]=n32,te[10]=n33,te[14]=n34,te[3]=n41,te[7]=n42,te[11]=n43,te[15]=n44,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Matrix4().fromArray(this.elements)}copy(m){const te=this.elements,me=m.elements;return te[0]=me[0],te[1]=me[1],te[2]=me[2],te[3]=me[3],te[4]=me[4],te[5]=me[5],te[6]=me[6],te[7]=me[7],te[8]=me[8],te[9]=me[9],te[10]=me[10],te[11]=me[11],te[12]=me[12],te[13]=me[13],te[14]=me[14],te[15]=me[15],this}copyPosition(m){const te=this.elements,me=m.elements;return te[12]=me[12],te[13]=me[13],te[14]=me[14],this}setFromMatrix3(m){const me=m.elements;return this.set(me[0],me[3],me[6],0,me[1],me[4],me[7],0,me[2],me[5],me[8],0,0,0,0,1),this}extractBasis(xAxis,yAxis,zAxis){return xAxis.setFromMatrixColumn(this,0),yAxis.setFromMatrixColumn(this,1),zAxis.setFromMatrixColumn(this,2),this}makeBasis(xAxis,yAxis,zAxis){return this.set(xAxis.x,yAxis.x,zAxis.x,0,xAxis.y,yAxis.y,zAxis.y,0,xAxis.z,yAxis.z,zAxis.z,0,0,0,0,1),this}extractRotation(m){const te=this.elements,me=m.elements,scaleX=1/_v1$5.setFromMatrixColumn(m,0).length(),scaleY=1/_v1$5.setFromMatrixColumn(m,1).length(),scaleZ=1/_v1$5.setFromMatrixColumn(m,2).length();return te[0]=me[0]*scaleX,te[1]=me[1]*scaleX,te[2]=me[2]*scaleX,te[3]=0,te[4]=me[4]*scaleY,te[5]=me[5]*scaleY,te[6]=me[6]*scaleY,te[7]=0,te[8]=me[8]*scaleZ,te[9]=me[9]*scaleZ,te[10]=me[10]*scaleZ,te[11]=0,te[12]=0,te[13]=0,te[14]=0,te[15]=1,this}makeRotationFromEuler(euler){const te=this.elements,x=euler.x,y=euler.y,z=euler.z,a=Math.cos(x),b=Math.sin(x),c=Math.cos(y),d=Math.sin(y),e=Math.cos(z),f=Math.sin(z);if(euler.order==="XYZ"){const ae=a*e,af=a*f,be=b*e,bf=b*f;te[0]=c*e,te[4]=-c*f,te[8]=d,te[1]=af+be*d,te[5]=ae-bf*d,te[9]=-b*c,te[2]=bf-ae*d,te[6]=be+af*d,te[10]=a*c}else if(euler.order==="YXZ"){const ce=c*e,cf=c*f,de=d*e,df=d*f;te[0]=ce+df*b,te[4]=de*b-cf,te[8]=a*d,te[1]=a*f,te[5]=a*e,te[9]=-b,te[2]=cf*b-de,te[6]=df+ce*b,te[10]=a*c}else if(euler.order==="ZXY"){const ce=c*e,cf=c*f,de=d*e,df=d*f;te[0]=ce-df*b,te[4]=-a*f,te[8]=de+cf*b,te[1]=cf+de*b,te[5]=a*e,te[9]=df-ce*b,te[2]=-a*d,te[6]=b,te[10]=a*c}else if(euler.order==="ZYX"){const ae=a*e,af=a*f,be=b*e,bf=b*f;te[0]=c*e,te[4]=be*d-af,te[8]=ae*d+bf,te[1]=c*f,te[5]=bf*d+ae,te[9]=af*d-be,te[2]=-d,te[6]=b*c,te[10]=a*c}else if(euler.order==="YZX"){const ac=a*c,ad=a*d,bc=b*c,bd=b*d;te[0]=c*e,te[4]=bd-ac*f,te[8]=bc*f+ad,te[1]=f,te[5]=a*e,te[9]=-b*e,te[2]=-d*e,te[6]=ad*f+bc,te[10]=ac-bd*f}else if(euler.order==="XZY"){const ac=a*c,ad=a*d,bc=b*c,bd=b*d;te[0]=c*e,te[4]=-f,te[8]=d*e,te[1]=ac*f+bd,te[5]=a*e,te[9]=ad*f-bc,te[2]=bc*f-ad,te[6]=b*e,te[10]=bd*f+ac}return te[3]=0,te[7]=0,te[11]=0,te[12]=0,te[13]=0,te[14]=0,te[15]=1,this}makeRotationFromQuaternion(q){return this.compose(_zero,q,_one)}lookAt(eye,target,up){const te=this.elements;return _z.subVectors(eye,target),_z.lengthSq()===0&&(_z.z=1),_z.normalize(),_x.crossVectors(up,_z),_x.lengthSq()===0&&(Math.abs(up.z)===1?_z.x+=1e-4:_z.z+=1e-4,_z.normalize(),_x.crossVectors(up,_z)),_x.normalize(),_y.crossVectors(_z,_x),te[0]=_x.x,te[4]=_y.x,te[8]=_z.x,te[1]=_x.y,te[5]=_y.y,te[9]=_z.y,te[2]=_x.z,te[6]=_y.z,te[10]=_z.z,this}multiply(m){return this.multiplyMatrices(this,m)}premultiply(m){return this.multiplyMatrices(m,this)}multiplyMatrices(a,b){const ae=a.elements,be=b.elements,te=this.elements,a11=ae[0],a12=ae[4],a13=ae[8],a14=ae[12],a21=ae[1],a22=ae[5],a23=ae[9],a24=ae[13],a31=ae[2],a32=ae[6],a33=ae[10],a34=ae[14],a41=ae[3],a42=ae[7],a43=ae[11],a44=ae[15],b11=be[0],b12=be[4],b13=be[8],b14=be[12],b21=be[1],b22=be[5],b23=be[9],b24=be[13],b31=be[2],b32=be[6],b33=be[10],b34=be[14],b41=be[3],b42=be[7],b43=be[11],b44=be[15];return te[0]=a11*b11+a12*b21+a13*b31+a14*b41,te[4]=a11*b12+a12*b22+a13*b32+a14*b42,te[8]=a11*b13+a12*b23+a13*b33+a14*b43,te[12]=a11*b14+a12*b24+a13*b34+a14*b44,te[1]=a21*b11+a22*b21+a23*b31+a24*b41,te[5]=a21*b12+a22*b22+a23*b32+a24*b42,te[9]=a21*b13+a22*b23+a23*b33+a24*b43,te[13]=a21*b14+a22*b24+a23*b34+a24*b44,te[2]=a31*b11+a32*b21+a33*b31+a34*b41,te[6]=a31*b12+a32*b22+a33*b32+a34*b42,te[10]=a31*b13+a32*b23+a33*b33+a34*b43,te[14]=a31*b14+a32*b24+a33*b34+a34*b44,te[3]=a41*b11+a42*b21+a43*b31+a44*b41,te[7]=a41*b12+a42*b22+a43*b32+a44*b42,te[11]=a41*b13+a42*b23+a43*b33+a44*b43,te[15]=a41*b14+a42*b24+a43*b34+a44*b44,this}multiplyScalar(s){const te=this.elements;return te[0]*=s,te[4]*=s,te[8]*=s,te[12]*=s,te[1]*=s,te[5]*=s,te[9]*=s,te[13]*=s,te[2]*=s,te[6]*=s,te[10]*=s,te[14]*=s,te[3]*=s,te[7]*=s,te[11]*=s,te[15]*=s,this}determinant(){const te=this.elements,n11=te[0],n12=te[4],n13=te[8],n14=te[12],n21=te[1],n22=te[5],n23=te[9],n24=te[13],n31=te[2],n32=te[6],n33=te[10],n34=te[14],n41=te[3],n42=te[7],n43=te[11],n44=te[15];return n41*(+n14*n23*n32-n13*n24*n32-n14*n22*n33+n12*n24*n33+n13*n22*n34-n12*n23*n34)+n42*(+n11*n23*n34-n11*n24*n33+n14*n21*n33-n13*n21*n34+n13*n24*n31-n14*n23*n31)+n43*(+n11*n24*n32-n11*n22*n34-n14*n21*n32+n12*n21*n34+n14*n22*n31-n12*n24*n31)+n44*(-n13*n22*n31-n11*n23*n32+n11*n22*n33+n13*n21*n32-n12*n21*n33+n12*n23*n31)}transpose(){const te=this.elements;let tmp;return tmp=te[1],te[1]=te[4],te[4]=tmp,tmp=te[2],te[2]=te[8],te[8]=tmp,tmp=te[6],te[6]=te[9],te[9]=tmp,tmp=te[3],te[3]=te[12],te[12]=tmp,tmp=te[7],te[7]=te[13],te[13]=tmp,tmp=te[11],te[11]=te[14],te[14]=tmp,this}setPosition(x,y,z){const te=this.elements;return x.isVector3?(te[12]=x.x,te[13]=x.y,te[14]=x.z):(te[12]=x,te[13]=y,te[14]=z),this}invert(){const te=this.elements,n11=te[0],n21=te[1],n31=te[2],n41=te[3],n12=te[4],n22=te[5],n32=te[6],n42=te[7],n13=te[8],n23=te[9],n33=te[10],n43=te[11],n14=te[12],n24=te[13],n34=te[14],n44=te[15],t11=n23*n34*n42-n24*n33*n42+n24*n32*n43-n22*n34*n43-n23*n32*n44+n22*n33*n44,t12=n14*n33*n42-n13*n34*n42-n14*n32*n43+n12*n34*n43+n13*n32*n44-n12*n33*n44,t13=n13*n24*n42-n14*n23*n42+n14*n22*n43-n12*n24*n43-n13*n22*n44+n12*n23*n44,t14=n14*n23*n32-n13*n24*n32-n14*n22*n33+n12*n24*n33+n13*n22*n34-n12*n23*n34,det=n11*t11+n21*t12+n31*t13+n41*t14;if(det===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const detInv=1/det;return te[0]=t11*detInv,te[1]=(n24*n33*n41-n23*n34*n41-n24*n31*n43+n21*n34*n43+n23*n31*n44-n21*n33*n44)*detInv,te[2]=(n22*n34*n41-n24*n32*n41+n24*n31*n42-n21*n34*n42-n22*n31*n44+n21*n32*n44)*detInv,te[3]=(n23*n32*n41-n22*n33*n41-n23*n31*n42+n21*n33*n42+n22*n31*n43-n21*n32*n43)*detInv,te[4]=t12*detInv,te[5]=(n13*n34*n41-n14*n33*n41+n14*n31*n43-n11*n34*n43-n13*n31*n44+n11*n33*n44)*detInv,te[6]=(n14*n32*n41-n12*n34*n41-n14*n31*n42+n11*n34*n42+n12*n31*n44-n11*n32*n44)*detInv,te[7]=(n12*n33*n41-n13*n32*n41+n13*n31*n42-n11*n33*n42-n12*n31*n43+n11*n32*n43)*detInv,te[8]=t13*detInv,te[9]=(n14*n23*n41-n13*n24*n41-n14*n21*n43+n11*n24*n43+n13*n21*n44-n11*n23*n44)*detInv,te[10]=(n12*n24*n41-n14*n22*n41+n14*n21*n42-n11*n24*n42-n12*n21*n44+n11*n22*n44)*detInv,te[11]=(n13*n22*n41-n12*n23*n41-n13*n21*n42+n11*n23*n42+n12*n21*n43-n11*n22*n43)*detInv,te[12]=t14*detInv,te[13]=(n13*n24*n31-n14*n23*n31+n14*n21*n33-n11*n24*n33-n13*n21*n34+n11*n23*n34)*detInv,te[14]=(n14*n22*n31-n12*n24*n31-n14*n21*n32+n11*n24*n32+n12*n21*n34-n11*n22*n34)*detInv,te[15]=(n12*n23*n31-n13*n22*n31+n13*n21*n32-n11*n23*n32-n12*n21*n33+n11*n22*n33)*detInv,this}scale(v){const te=this.elements,x=v.x,y=v.y,z=v.z;return te[0]*=x,te[4]*=y,te[8]*=z,te[1]*=x,te[5]*=y,te[9]*=z,te[2]*=x,te[6]*=y,te[10]*=z,te[3]*=x,te[7]*=y,te[11]*=z,this}getMaxScaleOnAxis(){const te=this.elements,scaleXSq=te[0]*te[0]+te[1]*te[1]+te[2]*te[2],scaleYSq=te[4]*te[4]+te[5]*te[5]+te[6]*te[6],scaleZSq=te[8]*te[8]+te[9]*te[9]+te[10]*te[10];return Math.sqrt(Math.max(scaleXSq,scaleYSq,scaleZSq))}makeTranslation(x,y,z){return x.isVector3?this.set(1,0,0,x.x,0,1,0,x.y,0,0,1,x.z,0,0,0,1):this.set(1,0,0,x,0,1,0,y,0,0,1,z,0,0,0,1),this}makeRotationX(theta){const c=Math.cos(theta),s=Math.sin(theta);return this.set(1,0,0,0,0,c,-s,0,0,s,c,0,0,0,0,1),this}makeRotationY(theta){const c=Math.cos(theta),s=Math.sin(theta);return this.set(c,0,s,0,0,1,0,0,-s,0,c,0,0,0,0,1),this}makeRotationZ(theta){const c=Math.cos(theta),s=Math.sin(theta);return this.set(c,-s,0,0,s,c,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(axis,angle){const c=Math.cos(angle),s=Math.sin(angle),t2=1-c,x=axis.x,y=axis.y,z=axis.z,tx=t2*x,ty=t2*y;return this.set(tx*x+c,tx*y-s*z,tx*z+s*y,0,tx*y+s*z,ty*y+c,ty*z-s*x,0,tx*z-s*y,ty*z+s*x,t2*z*z+c,0,0,0,0,1),this}makeScale(x,y,z){return this.set(x,0,0,0,0,y,0,0,0,0,z,0,0,0,0,1),this}makeShear(xy,xz,yx,yz,zx,zy){return this.set(1,yx,zx,0,xy,1,zy,0,xz,yz,1,0,0,0,0,1),this}compose(position,quaternion,scale){const te=this.elements,x=quaternion._x,y=quaternion._y,z=quaternion._z,w=quaternion._w,x2=x+x,y2=y+y,z2=z+z,xx=x*x2,xy=x*y2,xz=x*z2,yy=y*y2,yz=y*z2,zz=z*z2,wx=w*x2,wy=w*y2,wz=w*z2,sx=scale.x,sy=scale.y,sz=scale.z;return te[0]=(1-(yy+zz))*sx,te[1]=(xy+wz)*sx,te[2]=(xz-wy)*sx,te[3]=0,te[4]=(xy-wz)*sy,te[5]=(1-(xx+zz))*sy,te[6]=(yz+wx)*sy,te[7]=0,te[8]=(xz+wy)*sz,te[9]=(yz-wx)*sz,te[10]=(1-(xx+yy))*sz,te[11]=0,te[12]=position.x,te[13]=position.y,te[14]=position.z,te[15]=1,this}decompose(position,quaternion,scale){const te=this.elements;let sx=_v1$5.set(te[0],te[1],te[2]).length();const sy=_v1$5.set(te[4],te[5],te[6]).length(),sz=_v1$5.set(te[8],te[9],te[10]).length();this.determinant()<0&&(sx=-sx),position.x=te[12],position.y=te[13],position.z=te[14],_m1$4.copy(this);const invSX=1/sx,invSY=1/sy,invSZ=1/sz;return _m1$4.elements[0]*=invSX,_m1$4.elements[1]*=invSX,_m1$4.elements[2]*=invSX,_m1$4.elements[4]*=invSY,_m1$4.elements[5]*=invSY,_m1$4.elements[6]*=invSY,_m1$4.elements[8]*=invSZ,_m1$4.elements[9]*=invSZ,_m1$4.elements[10]*=invSZ,quaternion.setFromRotationMatrix(_m1$4),scale.x=sx,scale.y=sy,scale.z=sz,this}makePerspective(left,right,top,bottom,near,far,coordinateSystem=WebGLCoordinateSystem){const te=this.elements,x=2*near/(right-left),y=2*near/(top-bottom),a=(right+left)/(right-left),b=(top+bottom)/(top-bottom);let c,d;if(coordinateSystem===WebGLCoordinateSystem)c=-(far+near)/(far-near),d=-2*far*near/(far-near);else if(coordinateSystem===WebGPUCoordinateSystem)c=-far/(far-near),d=-far*near/(far-near);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+coordinateSystem);return te[0]=x,te[4]=0,te[8]=a,te[12]=0,te[1]=0,te[5]=y,te[9]=b,te[13]=0,te[2]=0,te[6]=0,te[10]=c,te[14]=d,te[3]=0,te[7]=0,te[11]=-1,te[15]=0,this}makeOrthographic(left,right,top,bottom,near,far,coordinateSystem=WebGLCoordinateSystem){const te=this.elements,w=1/(right-left),h=1/(top-bottom),p=1/(far-near),x=(right+left)*w,y=(top+bottom)*h;let z,zInv;if(coordinateSystem===WebGLCoordinateSystem)z=(far+near)*p,zInv=-2*p;else if(coordinateSystem===WebGPUCoordinateSystem)z=near*p,zInv=-1*p;else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+coordinateSystem);return te[0]=2*w,te[4]=0,te[8]=0,te[12]=-x,te[1]=0,te[5]=2*h,te[9]=0,te[13]=-y,te[2]=0,te[6]=0,te[10]=zInv,te[14]=-z,te[3]=0,te[7]=0,te[11]=0,te[15]=1,this}equals(matrix){const te=this.elements,me=matrix.elements;for(let i=0;i<16;i++)if(te[i]!==me[i])return!1;return!0}fromArray(array,offset=0){for(let i=0;i<16;i++)this.elements[i]=array[i+offset];return this}toArray(array=[],offset=0){const te=this.elements;return array[offset]=te[0],array[offset+1]=te[1],array[offset+2]=te[2],array[offset+3]=te[3],array[offset+4]=te[4],array[offset+5]=te[5],array[offset+6]=te[6],array[offset+7]=te[7],array[offset+8]=te[8],array[offset+9]=te[9],array[offset+10]=te[10],array[offset+11]=te[11],array[offset+12]=te[12],array[offset+13]=te[13],array[offset+14]=te[14],array[offset+15]=te[15],array}}const _v1$5=new Vector3,_m1$4=new Matrix4,_zero=new Vector3(0,0,0),_one=new Vector3(1,1,1),_x=new Vector3,_y=new Vector3,_z=new Vector3,_matrix$2=new Matrix4,_quaternion$3=new Quaternion;class Euler{static{__name(this,"Euler")}constructor(x=0,y=0,z=0,order=Euler.DEFAULT_ORDER){this.isEuler=!0,this._x=x,this._y=y,this._z=z,this._order=order}get x(){return this._x}set x(value){this._x=value,this._onChangeCallback()}get y(){return this._y}set y(value){this._y=value,this._onChangeCallback()}get z(){return this._z}set z(value){this._z=value,this._onChangeCallback()}get order(){return this._order}set order(value){this._order=value,this._onChangeCallback()}set(x,y,z,order=this._order){return this._x=x,this._y=y,this._z=z,this._order=order,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(euler){return this._x=euler._x,this._y=euler._y,this._z=euler._z,this._order=euler._order,this._onChangeCallback(),this}setFromRotationMatrix(m,order=this._order,update=!0){const te=m.elements,m11=te[0],m12=te[4],m13=te[8],m21=te[1],m22=te[5],m23=te[9],m31=te[2],m32=te[6],m33=te[10];switch(order){case"XYZ":this._y=Math.asin(clamp(m13,-1,1)),Math.abs(m13)<.9999999?(this._x=Math.atan2(-m23,m33),this._z=Math.atan2(-m12,m11)):(this._x=Math.atan2(m32,m22),this._z=0);break;case"YXZ":this._x=Math.asin(-clamp(m23,-1,1)),Math.abs(m23)<.9999999?(this._y=Math.atan2(m13,m33),this._z=Math.atan2(m21,m22)):(this._y=Math.atan2(-m31,m11),this._z=0);break;case"ZXY":this._x=Math.asin(clamp(m32,-1,1)),Math.abs(m32)<.9999999?(this._y=Math.atan2(-m31,m33),this._z=Math.atan2(-m12,m22)):(this._y=0,this._z=Math.atan2(m21,m11));break;case"ZYX":this._y=Math.asin(-clamp(m31,-1,1)),Math.abs(m31)<.9999999?(this._x=Math.atan2(m32,m33),this._z=Math.atan2(m21,m11)):(this._x=0,this._z=Math.atan2(-m12,m22));break;case"YZX":this._z=Math.asin(clamp(m21,-1,1)),Math.abs(m21)<.9999999?(this._x=Math.atan2(-m23,m22),this._y=Math.atan2(-m31,m11)):(this._x=0,this._y=Math.atan2(m13,m33));break;case"XZY":this._z=Math.asin(-clamp(m12,-1,1)),Math.abs(m12)<.9999999?(this._x=Math.atan2(m32,m22),this._y=Math.atan2(m13,m11)):(this._x=Math.atan2(-m23,m33),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+order)}return this._order=order,update===!0&&this._onChangeCallback(),this}setFromQuaternion(q,order,update){return _matrix$2.makeRotationFromQuaternion(q),this.setFromRotationMatrix(_matrix$2,order,update)}setFromVector3(v,order=this._order){return this.set(v.x,v.y,v.z,order)}reorder(newOrder){return _quaternion$3.setFromEuler(this),this.setFromQuaternion(_quaternion$3,newOrder)}equals(euler){return euler._x===this._x&&euler._y===this._y&&euler._z===this._z&&euler._order===this._order}fromArray(array){return this._x=array[0],this._y=array[1],this._z=array[2],array[3]!==void 0&&(this._order=array[3]),this._onChangeCallback(),this}toArray(array=[],offset=0){return array[offset]=this._x,array[offset+1]=this._y,array[offset+2]=this._z,array[offset+3]=this._order,array}_onChange(callback){return this._onChangeCallback=callback,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Euler.DEFAULT_ORDER="XYZ";class Layers{static{__name(this,"Layers")}constructor(){this.mask=1}set(channel){this.mask=(1<<channel|0)>>>0}enable(channel){this.mask|=1<<channel|0}enableAll(){this.mask=-1}toggle(channel){this.mask^=1<<channel|0}disable(channel){this.mask&=~(1<<channel|0)}disableAll(){this.mask=0}test(layers){return(this.mask&layers.mask)!==0}isEnabled(channel){return(this.mask&(1<<channel|0))!==0}}let _object3DId=0;const _v1$4=new Vector3,_q1=new Quaternion,_m1$3=new Matrix4,_target=new Vector3,_position$3=new Vector3,_scale$2=new Vector3,_quaternion$2=new Quaternion,_xAxis=new Vector3(1,0,0),_yAxis=new Vector3(0,1,0),_zAxis=new Vector3(0,0,1),_addedEvent={type:"added"},_removedEvent={type:"removed"},_childaddedEvent={type:"childadded",child:null},_childremovedEvent={type:"childremoved",child:null};class Object3D extends EventDispatcher{static{__name(this,"Object3D")}constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:_object3DId++}),this.uuid=generateUUID(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Object3D.DEFAULT_UP.clone();const position=new Vector3,rotation=new Euler,quaternion=new Quaternion,scale=new Vector3(1,1,1);function onRotationChange(){quaternion.setFromEuler(rotation,!1)}__name(onRotationChange,"onRotationChange");function onQuaternionChange(){rotation.setFromQuaternion(quaternion,void 0,!1)}__name(onQuaternionChange,"onQuaternionChange"),rotation._onChange(onRotationChange),quaternion._onChange(onQuaternionChange),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:position},rotation:{configurable:!0,enumerable:!0,value:rotation},quaternion:{configurable:!0,enumerable:!0,value:quaternion},scale:{configurable:!0,enumerable:!0,value:scale},modelViewMatrix:{value:new Matrix4},normalMatrix:{value:new Matrix3}}),this.matrix=new Matrix4,this.matrixWorld=new Matrix4,this.matrixAutoUpdate=Object3D.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Layers,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(matrix){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(matrix),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(q){return this.quaternion.premultiply(q),this}setRotationFromAxisAngle(axis,angle){this.quaternion.setFromAxisAngle(axis,angle)}setRotationFromEuler(euler){this.quaternion.setFromEuler(euler,!0)}setRotationFromMatrix(m){this.quaternion.setFromRotationMatrix(m)}setRotationFromQuaternion(q){this.quaternion.copy(q)}rotateOnAxis(axis,angle){return _q1.setFromAxisAngle(axis,angle),this.quaternion.multiply(_q1),this}rotateOnWorldAxis(axis,angle){return _q1.setFromAxisAngle(axis,angle),this.quaternion.premultiply(_q1),this}rotateX(angle){return this.rotateOnAxis(_xAxis,angle)}rotateY(angle){return this.rotateOnAxis(_yAxis,angle)}rotateZ(angle){return this.rotateOnAxis(_zAxis,angle)}translateOnAxis(axis,distance){return _v1$4.copy(axis).applyQuaternion(this.quaternion),this.position.add(_v1$4.multiplyScalar(distance)),this}translateX(distance){return this.translateOnAxis(_xAxis,distance)}translateY(distance){return this.translateOnAxis(_yAxis,distance)}translateZ(distance){return this.translateOnAxis(_zAxis,distance)}localToWorld(vector){return this.updateWorldMatrix(!0,!1),vector.applyMatrix4(this.matrixWorld)}worldToLocal(vector){return this.updateWorldMatrix(!0,!1),vector.applyMatrix4(_m1$3.copy(this.matrixWorld).invert())}lookAt(x,y,z){x.isVector3?_target.copy(x):_target.set(x,y,z);const parent=this.parent;this.updateWorldMatrix(!0,!1),_position$3.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?_m1$3.lookAt(_position$3,_target,this.up):_m1$3.lookAt(_target,_position$3,this.up),this.quaternion.setFromRotationMatrix(_m1$3),parent&&(_m1$3.extractRotation(parent.matrixWorld),_q1.setFromRotationMatrix(_m1$3),this.quaternion.premultiply(_q1.invert()))}add(object){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.add(arguments[i]);return this}return object===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",object),this):(object&&object.isObject3D?(object.removeFromParent(),object.parent=this,this.children.push(object),object.dispatchEvent(_addedEvent),_childaddedEvent.child=object,this.dispatchEvent(_childaddedEvent),_childaddedEvent.child=null):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",object),this)}remove(object){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}const index=this.children.indexOf(object);return index!==-1&&(object.parent=null,this.children.splice(index,1),object.dispatchEvent(_removedEvent),_childremovedEvent.child=object,this.dispatchEvent(_childremovedEvent),_childremovedEvent.child=null),this}removeFromParent(){const parent=this.parent;return parent!==null&&parent.remove(this),this}clear(){return this.remove(...this.children)}attach(object){return this.updateWorldMatrix(!0,!1),_m1$3.copy(this.matrixWorld).invert(),object.parent!==null&&(object.parent.updateWorldMatrix(!0,!1),_m1$3.multiply(object.parent.matrixWorld)),object.applyMatrix4(_m1$3),object.removeFromParent(),object.parent=this,this.children.push(object),object.updateWorldMatrix(!1,!0),object.dispatchEvent(_addedEvent),_childaddedEvent.child=object,this.dispatchEvent(_childaddedEvent),_childaddedEvent.child=null,this}getObjectById(id2){return this.getObjectByProperty("id",id2)}getObjectByName(name){return this.getObjectByProperty("name",name)}getObjectByProperty(name,value){if(this[name]===value)return this;for(let i=0,l=this.children.length;i<l;i++){const object=this.children[i].getObjectByProperty(name,value);if(object!==void 0)return object}}getObjectsByProperty(name,value,result=[]){this[name]===value&&result.push(this);const children=this.children;for(let i=0,l=children.length;i<l;i++)children[i].getObjectsByProperty(name,value,result);return result}getWorldPosition(target){return this.updateWorldMatrix(!0,!1),target.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(target){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(_position$3,target,_scale$2),target}getWorldScale(target){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(_position$3,_quaternion$2,target),target}getWorldDirection(target){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return target.set(e[8],e[9],e[10]).normalize()}raycast(){}traverse(callback){callback(this);const children=this.children;for(let i=0,l=children.length;i<l;i++)children[i].traverse(callback)}traverseVisible(callback){if(this.visible===!1)return;callback(this);const children=this.children;for(let i=0,l=children.length;i<l;i++)children[i].traverseVisible(callback)}traverseAncestors(callback){const parent=this.parent;parent!==null&&(callback(parent),parent.traverseAncestors(callback))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(force){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||force)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,force=!0);const children=this.children;for(let i=0,l=children.length;i<l;i++)children[i].updateMatrixWorld(force)}updateWorldMatrix(updateParents,updateChildren){const parent=this.parent;if(updateParents===!0&&parent!==null&&parent.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),updateChildren===!0){const children=this.children;for(let i=0,l=children.length;i<l;i++)children[i].updateWorldMatrix(!1,!0)}}toJSON(meta){const isRootObject=meta===void 0||typeof meta=="string",output={};isRootObject&&(meta={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},output.metadata={version:4.6,type:"Object",generator:"Object3D.toJSON"});const object={};object.uuid=this.uuid,object.type=this.type,this.name!==""&&(object.name=this.name),this.castShadow===!0&&(object.castShadow=!0),this.receiveShadow===!0&&(object.receiveShadow=!0),this.visible===!1&&(object.visible=!1),this.frustumCulled===!1&&(object.frustumCulled=!1),this.renderOrder!==0&&(object.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(object.userData=this.userData),object.layers=this.layers.mask,object.matrix=this.matrix.toArray(),object.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(object.matrixAutoUpdate=!1),this.isInstancedMesh&&(object.type="InstancedMesh",object.count=this.count,object.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(object.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(object.type="BatchedMesh",object.perObjectFrustumCulled=this.perObjectFrustumCulled,object.sortObjects=this.sortObjects,object.drawRanges=this._drawRanges,object.reservedRanges=this._reservedRanges,object.visibility=this._visibility,object.active=this._active,object.bounds=this._bounds.map(bound=>({boxInitialized:bound.boxInitialized,boxMin:bound.box.min.toArray(),boxMax:bound.box.max.toArray(),sphereInitialized:bound.sphereInitialized,sphereRadius:bound.sphere.radius,sphereCenter:bound.sphere.center.toArray()})),object.maxInstanceCount=this._maxInstanceCount,object.maxVertexCount=this._maxVertexCount,object.maxIndexCount=this._maxIndexCount,object.geometryInitialized=this._geometryInitialized,object.geometryCount=this._geometryCount,object.matricesTexture=this._matricesTexture.toJSON(meta),this._colorsTexture!==null&&(object.colorsTexture=this._colorsTexture.toJSON(meta)),this.boundingSphere!==null&&(object.boundingSphere={center:object.boundingSphere.center.toArray(),radius:object.boundingSphere.radius}),this.boundingBox!==null&&(object.boundingBox={min:object.boundingBox.min.toArray(),max:object.boundingBox.max.toArray()}));function serialize(library,element){return library[element.uuid]===void 0&&(library[element.uuid]=element.toJSON(meta)),element.uuid}if(__name(serialize,"serialize"),this.isScene)this.background&&(this.background.isColor?object.background=this.background.toJSON():this.background.isTexture&&(object.background=this.background.toJSON(meta).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(object.environment=this.environment.toJSON(meta).uuid);else if(this.isMesh||this.isLine||this.isPoints){object.geometry=serialize(meta.geometries,this.geometry);const parameters=this.geometry.parameters;if(parameters!==void 0&&parameters.shapes!==void 0){const shapes=parameters.shapes;if(Array.isArray(shapes))for(let i=0,l=shapes.length;i<l;i++){const shape=shapes[i];serialize(meta.shapes,shape)}else serialize(meta.shapes,shapes)}}if(this.isSkinnedMesh&&(object.bindMode=this.bindMode,object.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(serialize(meta.skeletons,this.skeleton),object.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const uuids=[];for(let i=0,l=this.material.length;i<l;i++)uuids.push(serialize(meta.materials,this.material[i]));object.material=uuids}else object.material=serialize(meta.materials,this.material);if(this.children.length>0){object.children=[];for(let i=0;i<this.children.length;i++)object.children.push(this.children[i].toJSON(meta).object)}if(this.animations.length>0){object.animations=[];for(let i=0;i<this.animations.length;i++){const animation=this.animations[i];object.animations.push(serialize(meta.animations,animation))}}if(isRootObject){const geometries=extractFromCache(meta.geometries),materials=extractFromCache(meta.materials),textures=extractFromCache(meta.textures),images=extractFromCache(meta.images),shapes=extractFromCache(meta.shapes),skeletons=extractFromCache(meta.skeletons),animations=extractFromCache(meta.animations),nodes=extractFromCache(meta.nodes);geometries.length>0&&(output.geometries=geometries),materials.length>0&&(output.materials=materials),textures.length>0&&(output.textures=textures),images.length>0&&(output.images=images),shapes.length>0&&(output.shapes=shapes),skeletons.length>0&&(output.skeletons=skeletons),animations.length>0&&(output.animations=animations),nodes.length>0&&(output.nodes=nodes)}return output.object=object,output;function extractFromCache(cache){const values=[];for(const key in cache){const data=cache[key];delete data.metadata,values.push(data)}return values}__name(extractFromCache,"extractFromCache")}clone(recursive){return new this.constructor().copy(this,recursive)}copy(source,recursive=!0){if(this.name=source.name,this.up.copy(source.up),this.position.copy(source.position),this.rotation.order=source.rotation.order,this.quaternion.copy(source.quaternion),this.scale.copy(source.scale),this.matrix.copy(source.matrix),this.matrixWorld.copy(source.matrixWorld),this.matrixAutoUpdate=source.matrixAutoUpdate,this.matrixWorldAutoUpdate=source.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=source.matrixWorldNeedsUpdate,this.layers.mask=source.layers.mask,this.visible=source.visible,this.castShadow=source.castShadow,this.receiveShadow=source.receiveShadow,this.frustumCulled=source.frustumCulled,this.renderOrder=source.renderOrder,this.animations=source.animations.slice(),this.userData=JSON.parse(JSON.stringify(source.userData)),recursive===!0)for(let i=0;i<source.children.length;i++){const child=source.children[i];this.add(child.clone())}return this}}Object3D.DEFAULT_UP=new Vector3(0,1,0);Object3D.DEFAULT_MATRIX_AUTO_UPDATE=!0;Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const _v0$2=new Vector3,_v1$3=new Vector3,_v2$2=new Vector3,_v3$2=new Vector3,_vab=new Vector3,_vac=new Vector3,_vbc=new Vector3,_vap=new Vector3,_vbp=new Vector3,_vcp=new Vector3,_v40=new Vector4,_v41=new Vector4,_v42=new Vector4;class Triangle{static{__name(this,"Triangle")}constructor(a=new Vector3,b=new Vector3,c=new Vector3){this.a=a,this.b=b,this.c=c}static getNormal(a,b,c,target){target.subVectors(c,b),_v0$2.subVectors(a,b),target.cross(_v0$2);const targetLengthSq=target.lengthSq();return targetLengthSq>0?target.multiplyScalar(1/Math.sqrt(targetLengthSq)):target.set(0,0,0)}static getBarycoord(point,a,b,c,target){_v0$2.subVectors(c,a),_v1$3.subVectors(b,a),_v2$2.subVectors(point,a);const dot00=_v0$2.dot(_v0$2),dot01=_v0$2.dot(_v1$3),dot02=_v0$2.dot(_v2$2),dot11=_v1$3.dot(_v1$3),dot12=_v1$3.dot(_v2$2),denom=dot00*dot11-dot01*dot01;if(denom===0)return target.set(0,0,0),null;const invDenom=1/denom,u=(dot11*dot02-dot01*dot12)*invDenom,v=(dot00*dot12-dot01*dot02)*invDenom;return target.set(1-u-v,v,u)}static containsPoint(point,a,b,c){return this.getBarycoord(point,a,b,c,_v3$2)===null?!1:_v3$2.x>=0&&_v3$2.y>=0&&_v3$2.x+_v3$2.y<=1}static getInterpolation(point,p1,p2,p3,v1,v2,v3,target){return this.getBarycoord(point,p1,p2,p3,_v3$2)===null?(target.x=0,target.y=0,"z"in target&&(target.z=0),"w"in target&&(target.w=0),null):(target.setScalar(0),target.addScaledVector(v1,_v3$2.x),target.addScaledVector(v2,_v3$2.y),target.addScaledVector(v3,_v3$2.z),target)}static getInterpolatedAttribute(attr,i1,i2,i3,barycoord,target){return _v40.setScalar(0),_v41.setScalar(0),_v42.setScalar(0),_v40.fromBufferAttribute(attr,i1),_v41.fromBufferAttribute(attr,i2),_v42.fromBufferAttribute(attr,i3),target.setScalar(0),target.addScaledVector(_v40,barycoord.x),target.addScaledVector(_v41,barycoord.y),target.addScaledVector(_v42,barycoord.z),target}static isFrontFacing(a,b,c,direction){return _v0$2.subVectors(c,b),_v1$3.subVectors(a,b),_v0$2.cross(_v1$3).dot(direction)<0}set(a,b,c){return this.a.copy(a),this.b.copy(b),this.c.copy(c),this}setFromPointsAndIndices(points,i0,i1,i2){return this.a.copy(points[i0]),this.b.copy(points[i1]),this.c.copy(points[i2]),this}setFromAttributeAndIndices(attribute,i0,i1,i2){return this.a.fromBufferAttribute(attribute,i0),this.b.fromBufferAttribute(attribute,i1),this.c.fromBufferAttribute(attribute,i2),this}clone(){return new this.constructor().copy(this)}copy(triangle){return this.a.copy(triangle.a),this.b.copy(triangle.b),this.c.copy(triangle.c),this}getArea(){return _v0$2.subVectors(this.c,this.b),_v1$3.subVectors(this.a,this.b),_v0$2.cross(_v1$3).length()*.5}getMidpoint(target){return target.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(target){return Triangle.getNormal(this.a,this.b,this.c,target)}getPlane(target){return target.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(point,target){return Triangle.getBarycoord(point,this.a,this.b,this.c,target)}getInterpolation(point,v1,v2,v3,target){return Triangle.getInterpolation(point,this.a,this.b,this.c,v1,v2,v3,target)}containsPoint(point){return Triangle.containsPoint(point,this.a,this.b,this.c)}isFrontFacing(direction){return Triangle.isFrontFacing(this.a,this.b,this.c,direction)}intersectsBox(box){return box.intersectsTriangle(this)}closestPointToPoint(p,target){const a=this.a,b=this.b,c=this.c;let v,w;_vab.subVectors(b,a),_vac.subVectors(c,a),_vap.subVectors(p,a);const d1=_vab.dot(_vap),d2=_vac.dot(_vap);if(d1<=0&&d2<=0)return target.copy(a);_vbp.subVectors(p,b);const d3=_vab.dot(_vbp),d4=_vac.dot(_vbp);if(d3>=0&&d4<=d3)return target.copy(b);const vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return v=d1/(d1-d3),target.copy(a).addScaledVector(_vab,v);_vcp.subVectors(p,c);const d5=_vab.dot(_vcp),d6=_vac.dot(_vcp);if(d6>=0&&d5<=d6)return target.copy(c);const vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return w=d2/(d2-d6),target.copy(a).addScaledVector(_vac,w);const va=d3*d6-d5*d4;if(va<=0&&d4-d3>=0&&d5-d6>=0)return _vbc.subVectors(c,b),w=(d4-d3)/(d4-d3+(d5-d6)),target.copy(b).addScaledVector(_vbc,w);const denom=1/(va+vb+vc);return v=vb*denom,w=vc*denom,target.copy(a).addScaledVector(_vab,v).addScaledVector(_vac,w)}equals(triangle){return triangle.a.equals(this.a)&&triangle.b.equals(this.b)&&triangle.c.equals(this.c)}}const _colorKeywords={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},_hslA={h:0,s:0,l:0},_hslB={h:0,s:0,l:0};function hue2rgb(p,q,t2){return t2<0&&(t2+=1),t2>1&&(t2-=1),t2<1/6?p+(q-p)*6*t2:t2<1/2?q:t2<2/3?p+(q-p)*6*(2/3-t2):p}__name(hue2rgb,"hue2rgb");class Color{static{__name(this,"Color")}constructor(r,g,b){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(r,g,b)}set(r,g,b){if(g===void 0&&b===void 0){const value=r;value&&value.isColor?this.copy(value):typeof value=="number"?this.setHex(value):typeof value=="string"&&this.setStyle(value)}else this.setRGB(r,g,b);return this}setScalar(scalar){return this.r=scalar,this.g=scalar,this.b=scalar,this}setHex(hex,colorSpace=SRGBColorSpace){return hex=Math.floor(hex),this.r=(hex>>16&255)/255,this.g=(hex>>8&255)/255,this.b=(hex&255)/255,ColorManagement.toWorkingColorSpace(this,colorSpace),this}setRGB(r,g,b,colorSpace=ColorManagement.workingColorSpace){return this.r=r,this.g=g,this.b=b,ColorManagement.toWorkingColorSpace(this,colorSpace),this}setHSL(h,s,l,colorSpace=ColorManagement.workingColorSpace){if(h=euclideanModulo(h,1),s=clamp(s,0,1),l=clamp(l,0,1),s===0)this.r=this.g=this.b=l;else{const p=l<=.5?l*(1+s):l+s-l*s,q=2*l-p;this.r=hue2rgb(q,p,h+1/3),this.g=hue2rgb(q,p,h),this.b=hue2rgb(q,p,h-1/3)}return ColorManagement.toWorkingColorSpace(this,colorSpace),this}setStyle(style,colorSpace=SRGBColorSpace){function handleAlpha(string){string!==void 0&&parseFloat(string)<1&&console.warn("THREE.Color: Alpha component of "+style+" will be ignored.")}__name(handleAlpha,"handleAlpha");let m;if(m=/^(\w+)\(([^\)]*)\)/.exec(style)){let color;const name=m[1],components=m[2];switch(name){case"rgb":case"rgba":if(color=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(components))return handleAlpha(color[4]),this.setRGB(Math.min(255,parseInt(color[1],10))/255,Math.min(255,parseInt(color[2],10))/255,Math.min(255,parseInt(color[3],10))/255,colorSpace);if(color=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(components))return handleAlpha(color[4]),this.setRGB(Math.min(100,parseInt(color[1],10))/100,Math.min(100,parseInt(color[2],10))/100,Math.min(100,parseInt(color[3],10))/100,colorSpace);break;case"hsl":case"hsla":if(color=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(components))return handleAlpha(color[4]),this.setHSL(parseFloat(color[1])/360,parseFloat(color[2])/100,parseFloat(color[3])/100,colorSpace);break;default:console.warn("THREE.Color: Unknown color model "+style)}}else if(m=/^\#([A-Fa-f\d]+)$/.exec(style)){const hex=m[1],size=hex.length;if(size===3)return this.setRGB(parseInt(hex.charAt(0),16)/15,parseInt(hex.charAt(1),16)/15,parseInt(hex.charAt(2),16)/15,colorSpace);if(size===6)return this.setHex(parseInt(hex,16),colorSpace);console.warn("THREE.Color: Invalid hex color "+style)}else if(style&&style.length>0)return this.setColorName(style,colorSpace);return this}setColorName(style,colorSpace=SRGBColorSpace){const hex=_colorKeywords[style.toLowerCase()];return hex!==void 0?this.setHex(hex,colorSpace):console.warn("THREE.Color: Unknown color "+style),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(color){return this.r=color.r,this.g=color.g,this.b=color.b,this}copySRGBToLinear(color){return this.r=SRGBToLinear(color.r),this.g=SRGBToLinear(color.g),this.b=SRGBToLinear(color.b),this}copyLinearToSRGB(color){return this.r=LinearToSRGB(color.r),this.g=LinearToSRGB(color.g),this.b=LinearToSRGB(color.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(colorSpace=SRGBColorSpace){return ColorManagement.fromWorkingColorSpace(_color$1.copy(this),colorSpace),Math.round(clamp(_color$1.r*255,0,255))*65536+Math.round(clamp(_color$1.g*255,0,255))*256+Math.round(clamp(_color$1.b*255,0,255))}getHexString(colorSpace=SRGBColorSpace){return("000000"+this.getHex(colorSpace).toString(16)).slice(-6)}getHSL(target,colorSpace=ColorManagement.workingColorSpace){ColorManagement.fromWorkingColorSpace(_color$1.copy(this),colorSpace);const r=_color$1.r,g=_color$1.g,b=_color$1.b,max2=Math.max(r,g,b),min=Math.min(r,g,b);let hue,saturation;const lightness=(min+max2)/2;if(min===max2)hue=0,saturation=0;else{const delta=max2-min;switch(saturation=lightness<=.5?delta/(max2+min):delta/(2-max2-min),max2){case r:hue=(g-b)/delta+(g<b?6:0);break;case g:hue=(b-r)/delta+2;break;case b:hue=(r-g)/delta+4;break}hue/=6}return target.h=hue,target.s=saturation,target.l=lightness,target}getRGB(target,colorSpace=ColorManagement.workingColorSpace){return ColorManagement.fromWorkingColorSpace(_color$1.copy(this),colorSpace),target.r=_color$1.r,target.g=_color$1.g,target.b=_color$1.b,target}getStyle(colorSpace=SRGBColorSpace){ColorManagement.fromWorkingColorSpace(_color$1.copy(this),colorSpace);const r=_color$1.r,g=_color$1.g,b=_color$1.b;return colorSpace!==SRGBColorSpace?`color(${colorSpace} ${r.toFixed(3)} ${g.toFixed(3)} ${b.toFixed(3)})`:`rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`}offsetHSL(h,s,l){return this.getHSL(_hslA),this.setHSL(_hslA.h+h,_hslA.s+s,_hslA.l+l)}add(color){return this.r+=color.r,this.g+=color.g,this.b+=color.b,this}addColors(color1,color2){return this.r=color1.r+color2.r,this.g=color1.g+color2.g,this.b=color1.b+color2.b,this}addScalar(s){return this.r+=s,this.g+=s,this.b+=s,this}sub(color){return this.r=Math.max(0,this.r-color.r),this.g=Math.max(0,this.g-color.g),this.b=Math.max(0,this.b-color.b),this}multiply(color){return this.r*=color.r,this.g*=color.g,this.b*=color.b,this}multiplyScalar(s){return this.r*=s,this.g*=s,this.b*=s,this}lerp(color,alpha){return this.r+=(color.r-this.r)*alpha,this.g+=(color.g-this.g)*alpha,this.b+=(color.b-this.b)*alpha,this}lerpColors(color1,color2,alpha){return this.r=color1.r+(color2.r-color1.r)*alpha,this.g=color1.g+(color2.g-color1.g)*alpha,this.b=color1.b+(color2.b-color1.b)*alpha,this}lerpHSL(color,alpha){this.getHSL(_hslA),color.getHSL(_hslB);const h=lerp(_hslA.h,_hslB.h,alpha),s=lerp(_hslA.s,_hslB.s,alpha),l=lerp(_hslA.l,_hslB.l,alpha);return this.setHSL(h,s,l),this}setFromVector3(v){return this.r=v.x,this.g=v.y,this.b=v.z,this}applyMatrix3(m){const r=this.r,g=this.g,b=this.b,e=m.elements;return this.r=e[0]*r+e[3]*g+e[6]*b,this.g=e[1]*r+e[4]*g+e[7]*b,this.b=e[2]*r+e[5]*g+e[8]*b,this}equals(c){return c.r===this.r&&c.g===this.g&&c.b===this.b}fromArray(array,offset=0){return this.r=array[offset],this.g=array[offset+1],this.b=array[offset+2],this}toArray(array=[],offset=0){return array[offset]=this.r,array[offset+1]=this.g,array[offset+2]=this.b,array}fromBufferAttribute(attribute,index){return this.r=attribute.getX(index),this.g=attribute.getY(index),this.b=attribute.getZ(index),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const _color$1=new Color;Color.NAMES=_colorKeywords;let _materialId=0;class Material extends EventDispatcher{static{__name(this,"Material")}static get type(){return"Material"}get type(){return this.constructor.type}set type(_value){}constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:_materialId++}),this.uuid=generateUUID(),this.name="",this.blending=NormalBlending,this.side=FrontSide,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=SrcAlphaFactor,this.blendDst=OneMinusSrcAlphaFactor,this.blendEquation=AddEquation,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Color(0,0,0),this.blendAlpha=0,this.depthFunc=LessEqualDepth,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=AlwaysStencilFunc,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=KeepStencilOp,this.stencilZFail=KeepStencilOp,this.stencilZPass=KeepStencilOp,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(value){this._alphaTest>0!=value>0&&this.version++,this._alphaTest=value}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(values){if(values!==void 0)for(const key in values){const newValue=values[key];if(newValue===void 0){console.warn(`THREE.Material: parameter '${key}' has value of undefined.`);continue}const currentValue=this[key];if(currentValue===void 0){console.warn(`THREE.Material: '${key}' is not a property of THREE.${this.type}.`);continue}currentValue&&currentValue.isColor?currentValue.set(newValue):currentValue&&currentValue.isVector3&&newValue&&newValue.isVector3?currentValue.copy(newValue):this[key]=newValue}}toJSON(meta){const isRootObject=meta===void 0||typeof meta=="string";isRootObject&&(meta={textures:{},images:{}});const data={metadata:{version:4.6,type:"Material",generator:"Material.toJSON"}};data.uuid=this.uuid,data.type=this.type,this.name!==""&&(data.name=this.name),this.color&&this.color.isColor&&(data.color=this.color.getHex()),this.roughness!==void 0&&(data.roughness=this.roughness),this.metalness!==void 0&&(data.metalness=this.metalness),this.sheen!==void 0&&(data.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(data.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(data.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(data.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(data.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(data.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(data.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(data.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(data.shininess=this.shininess),this.clearcoat!==void 0&&(data.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(data.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(data.clearcoatMap=this.clearcoatMap.toJSON(meta).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(data.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(meta).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(data.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(meta).uuid,data.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.dispersion!==void 0&&(data.dispersion=this.dispersion),this.iridescence!==void 0&&(data.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(data.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(data.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(data.iridescenceMap=this.iridescenceMap.toJSON(meta).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(data.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(meta).uuid),this.anisotropy!==void 0&&(data.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(data.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(data.anisotropyMap=this.anisotropyMap.toJSON(meta).uuid),this.map&&this.map.isTexture&&(data.map=this.map.toJSON(meta).uuid),this.matcap&&this.matcap.isTexture&&(data.matcap=this.matcap.toJSON(meta).uuid),this.alphaMap&&this.alphaMap.isTexture&&(data.alphaMap=this.alphaMap.toJSON(meta).uuid),this.lightMap&&this.lightMap.isTexture&&(data.lightMap=this.lightMap.toJSON(meta).uuid,data.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(data.aoMap=this.aoMap.toJSON(meta).uuid,data.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(data.bumpMap=this.bumpMap.toJSON(meta).uuid,data.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(data.normalMap=this.normalMap.toJSON(meta).uuid,data.normalMapType=this.normalMapType,data.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(data.displacementMap=this.displacementMap.toJSON(meta).uuid,data.displacementScale=this.displacementScale,data.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(data.roughnessMap=this.roughnessMap.toJSON(meta).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(data.metalnessMap=this.metalnessMap.toJSON(meta).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(data.emissiveMap=this.emissiveMap.toJSON(meta).uuid),this.specularMap&&this.specularMap.isTexture&&(data.specularMap=this.specularMap.toJSON(meta).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(data.specularIntensityMap=this.specularIntensityMap.toJSON(meta).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(data.specularColorMap=this.specularColorMap.toJSON(meta).uuid),this.envMap&&this.envMap.isTexture&&(data.envMap=this.envMap.toJSON(meta).uuid,this.combine!==void 0&&(data.combine=this.combine)),this.envMapRotation!==void 0&&(data.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(data.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(data.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(data.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(data.gradientMap=this.gradientMap.toJSON(meta).uuid),this.transmission!==void 0&&(data.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(data.transmissionMap=this.transmissionMap.toJSON(meta).uuid),this.thickness!==void 0&&(data.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(data.thicknessMap=this.thicknessMap.toJSON(meta).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(data.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(data.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(data.size=this.size),this.shadowSide!==null&&(data.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(data.sizeAttenuation=this.sizeAttenuation),this.blending!==NormalBlending&&(data.blending=this.blending),this.side!==FrontSide&&(data.side=this.side),this.vertexColors===!0&&(data.vertexColors=!0),this.opacity<1&&(data.opacity=this.opacity),this.transparent===!0&&(data.transparent=!0),this.blendSrc!==SrcAlphaFactor&&(data.blendSrc=this.blendSrc),this.blendDst!==OneMinusSrcAlphaFactor&&(data.blendDst=this.blendDst),this.blendEquation!==AddEquation&&(data.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(data.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(data.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(data.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(data.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(data.blendAlpha=this.blendAlpha),this.depthFunc!==LessEqualDepth&&(data.depthFunc=this.depthFunc),this.depthTest===!1&&(data.depthTest=this.depthTest),this.depthWrite===!1&&(data.depthWrite=this.depthWrite),this.colorWrite===!1&&(data.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(data.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==AlwaysStencilFunc&&(data.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(data.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(data.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==KeepStencilOp&&(data.stencilFail=this.stencilFail),this.stencilZFail!==KeepStencilOp&&(data.stencilZFail=this.stencilZFail),this.stencilZPass!==KeepStencilOp&&(data.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(data.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(data.rotation=this.rotation),this.polygonOffset===!0&&(data.polygonOffset=!0),this.polygonOffsetFactor!==0&&(data.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(data.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(data.linewidth=this.linewidth),this.dashSize!==void 0&&(data.dashSize=this.dashSize),this.gapSize!==void 0&&(data.gapSize=this.gapSize),this.scale!==void 0&&(data.scale=this.scale),this.dithering===!0&&(data.dithering=!0),this.alphaTest>0&&(data.alphaTest=this.alphaTest),this.alphaHash===!0&&(data.alphaHash=!0),this.alphaToCoverage===!0&&(data.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(data.premultipliedAlpha=!0),this.forceSinglePass===!0&&(data.forceSinglePass=!0),this.wireframe===!0&&(data.wireframe=!0),this.wireframeLinewidth>1&&(data.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(data.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(data.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(data.flatShading=!0),this.visible===!1&&(data.visible=!1),this.toneMapped===!1&&(data.toneMapped=!1),this.fog===!1&&(data.fog=!1),Object.keys(this.userData).length>0&&(data.userData=this.userData);function extractFromCache(cache){const values=[];for(const key in cache){const data2=cache[key];delete data2.metadata,values.push(data2)}return values}if(__name(extractFromCache,"extractFromCache"),isRootObject){const textures=extractFromCache(meta.textures),images=extractFromCache(meta.images);textures.length>0&&(data.textures=textures),images.length>0&&(data.images=images)}return data}clone(){return new this.constructor().copy(this)}copy(source){this.name=source.name,this.blending=source.blending,this.side=source.side,this.vertexColors=source.vertexColors,this.opacity=source.opacity,this.transparent=source.transparent,this.blendSrc=source.blendSrc,this.blendDst=source.blendDst,this.blendEquation=source.blendEquation,this.blendSrcAlpha=source.blendSrcAlpha,this.blendDstAlpha=source.blendDstAlpha,this.blendEquationAlpha=source.blendEquationAlpha,this.blendColor.copy(source.blendColor),this.blendAlpha=source.blendAlpha,this.depthFunc=source.depthFunc,this.depthTest=source.depthTest,this.depthWrite=source.depthWrite,this.stencilWriteMask=source.stencilWriteMask,this.stencilFunc=source.stencilFunc,this.stencilRef=source.stencilRef,this.stencilFuncMask=source.stencilFuncMask,this.stencilFail=source.stencilFail,this.stencilZFail=source.stencilZFail,this.stencilZPass=source.stencilZPass,this.stencilWrite=source.stencilWrite;const srcPlanes=source.clippingPlanes;let dstPlanes=null;if(srcPlanes!==null){const n=srcPlanes.length;dstPlanes=new Array(n);for(let i=0;i!==n;++i)dstPlanes[i]=srcPlanes[i].clone()}return this.clippingPlanes=dstPlanes,this.clipIntersection=source.clipIntersection,this.clipShadows=source.clipShadows,this.shadowSide=source.shadowSide,this.colorWrite=source.colorWrite,this.precision=source.precision,this.polygonOffset=source.polygonOffset,this.polygonOffsetFactor=source.polygonOffsetFactor,this.polygonOffsetUnits=source.polygonOffsetUnits,this.dithering=source.dithering,this.alphaTest=source.alphaTest,this.alphaHash=source.alphaHash,this.alphaToCoverage=source.alphaToCoverage,this.premultipliedAlpha=source.premultipliedAlpha,this.forceSinglePass=source.forceSinglePass,this.visible=source.visible,this.toneMapped=source.toneMapped,this.userData=JSON.parse(JSON.stringify(source.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(value){value===!0&&this.version++}onBuild(){console.warn("Material: onBuild() has been removed.")}}class MeshBasicMaterial extends Material{static{__name(this,"MeshBasicMaterial")}static get type(){return"MeshBasicMaterial"}constructor(parameters){super(),this.isMeshBasicMaterial=!0,this.color=new Color(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Euler,this.combine=MultiplyOperation,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(parameters)}copy(source){return super.copy(source),this.color.copy(source.color),this.map=source.map,this.lightMap=source.lightMap,this.lightMapIntensity=source.lightMapIntensity,this.aoMap=source.aoMap,this.aoMapIntensity=source.aoMapIntensity,this.specularMap=source.specularMap,this.alphaMap=source.alphaMap,this.envMap=source.envMap,this.envMapRotation.copy(source.envMapRotation),this.combine=source.combine,this.reflectivity=source.reflectivity,this.refractionRatio=source.refractionRatio,this.wireframe=source.wireframe,this.wireframeLinewidth=source.wireframeLinewidth,this.wireframeLinecap=source.wireframeLinecap,this.wireframeLinejoin=source.wireframeLinejoin,this.fog=source.fog,this}}const _vector$9=new Vector3,_vector2$1=new Vector2;class BufferAttribute{static{__name(this,"BufferAttribute")}constructor(array,itemSize,normalized=!1){if(Array.isArray(array))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,this.name="",this.array=array,this.itemSize=itemSize,this.count=array!==void 0?array.length/itemSize:0,this.normalized=normalized,this.usage=StaticDrawUsage,this.updateRanges=[],this.gpuType=FloatType,this.version=0}onUploadCallback(){}set needsUpdate(value){value===!0&&this.version++}setUsage(value){return this.usage=value,this}addUpdateRange(start,count){this.updateRanges.push({start,count})}clearUpdateRanges(){this.updateRanges.length=0}copy(source){return this.name=source.name,this.array=new source.array.constructor(source.array),this.itemSize=source.itemSize,this.count=source.count,this.normalized=source.normalized,this.usage=source.usage,this.gpuType=source.gpuType,this}copyAt(index1,attribute,index2){index1*=this.itemSize,index2*=attribute.itemSize;for(let i=0,l=this.itemSize;i<l;i++)this.array[index1+i]=attribute.array[index2+i];return this}copyArray(array){return this.array.set(array),this}applyMatrix3(m){if(this.itemSize===2)for(let i=0,l=this.count;i<l;i++)_vector2$1.fromBufferAttribute(this,i),_vector2$1.applyMatrix3(m),this.setXY(i,_vector2$1.x,_vector2$1.y);else if(this.itemSize===3)for(let i=0,l=this.count;i<l;i++)_vector$9.fromBufferAttribute(this,i),_vector$9.applyMatrix3(m),this.setXYZ(i,_vector$9.x,_vector$9.y,_vector$9.z);return this}applyMatrix4(m){for(let i=0,l=this.count;i<l;i++)_vector$9.fromBufferAttribute(this,i),_vector$9.applyMatrix4(m),this.setXYZ(i,_vector$9.x,_vector$9.y,_vector$9.z);return this}applyNormalMatrix(m){for(let i=0,l=this.count;i<l;i++)_vector$9.fromBufferAttribute(this,i),_vector$9.applyNormalMatrix(m),this.setXYZ(i,_vector$9.x,_vector$9.y,_vector$9.z);return this}transformDirection(m){for(let i=0,l=this.count;i<l;i++)_vector$9.fromBufferAttribute(this,i),_vector$9.transformDirection(m),this.setXYZ(i,_vector$9.x,_vector$9.y,_vector$9.z);return this}set(value,offset=0){return this.array.set(value,offset),this}getComponent(index,component){let value=this.array[index*this.itemSize+component];return this.normalized&&(value=denormalize(value,this.array)),value}setComponent(index,component,value){return this.normalized&&(value=normalize(value,this.array)),this.array[index*this.itemSize+component]=value,this}getX(index){let x=this.array[index*this.itemSize];return this.normalized&&(x=denormalize(x,this.array)),x}setX(index,x){return this.normalized&&(x=normalize(x,this.array)),this.array[index*this.itemSize]=x,this}getY(index){let y=this.array[index*this.itemSize+1];return this.normalized&&(y=denormalize(y,this.array)),y}setY(index,y){return this.normalized&&(y=normalize(y,this.array)),this.array[index*this.itemSize+1]=y,this}getZ(index){let z=this.array[index*this.itemSize+2];return this.normalized&&(z=denormalize(z,this.array)),z}setZ(index,z){return this.normalized&&(z=normalize(z,this.array)),this.array[index*this.itemSize+2]=z,this}getW(index){let w=this.array[index*this.itemSize+3];return this.normalized&&(w=denormalize(w,this.array)),w}setW(index,w){return this.normalized&&(w=normalize(w,this.array)),this.array[index*this.itemSize+3]=w,this}setXY(index,x,y){return index*=this.itemSize,this.normalized&&(x=normalize(x,this.array),y=normalize(y,this.array)),this.array[index+0]=x,this.array[index+1]=y,this}setXYZ(index,x,y,z){return index*=this.itemSize,this.normalized&&(x=normalize(x,this.array),y=normalize(y,this.array),z=normalize(z,this.array)),this.array[index+0]=x,this.array[index+1]=y,this.array[index+2]=z,this}setXYZW(index,x,y,z,w){return index*=this.itemSize,this.normalized&&(x=normalize(x,this.array),y=normalize(y,this.array),z=normalize(z,this.array),w=normalize(w,this.array)),this.array[index+0]=x,this.array[index+1]=y,this.array[index+2]=z,this.array[index+3]=w,this}onUpload(callback){return this.onUploadCallback=callback,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const data={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(data.name=this.name),this.usage!==StaticDrawUsage&&(data.usage=this.usage),data}}class Uint16BufferAttribute extends BufferAttribute{static{__name(this,"Uint16BufferAttribute")}constructor(array,itemSize,normalized){super(new Uint16Array(array),itemSize,normalized)}}class Uint32BufferAttribute extends BufferAttribute{static{__name(this,"Uint32BufferAttribute")}constructor(array,itemSize,normalized){super(new Uint32Array(array),itemSize,normalized)}}class Float32BufferAttribute extends BufferAttribute{static{__name(this,"Float32BufferAttribute")}constructor(array,itemSize,normalized){super(new Float32Array(array),itemSize,normalized)}}let _id$2=0;const _m1$2=new Matrix4,_obj=new Object3D,_offset=new Vector3,_box$2=new Box3,_boxMorphTargets=new Box3,_vector$8=new Vector3;class BufferGeometry extends EventDispatcher{static{__name(this,"BufferGeometry")}constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:_id$2++}),this.uuid=generateUUID(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(index){return Array.isArray(index)?this.index=new(arrayNeedsUint32(index)?Uint32BufferAttribute:Uint16BufferAttribute)(index,1):this.index=index,this}setIndirect(indirect){return this.indirect=indirect,this}getIndirect(){return this.indirect}getAttribute(name){return this.attributes[name]}setAttribute(name,attribute){return this.attributes[name]=attribute,this}deleteAttribute(name){return delete this.attributes[name],this}hasAttribute(name){return this.attributes[name]!==void 0}addGroup(start,count,materialIndex=0){this.groups.push({start,count,materialIndex})}clearGroups(){this.groups=[]}setDrawRange(start,count){this.drawRange.start=start,this.drawRange.count=count}applyMatrix4(matrix){const position=this.attributes.position;position!==void 0&&(position.applyMatrix4(matrix),position.needsUpdate=!0);const normal=this.attributes.normal;if(normal!==void 0){const normalMatrix=new Matrix3().getNormalMatrix(matrix);normal.applyNormalMatrix(normalMatrix),normal.needsUpdate=!0}const tangent=this.attributes.tangent;return tangent!==void 0&&(tangent.transformDirection(matrix),tangent.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(q){return _m1$2.makeRotationFromQuaternion(q),this.applyMatrix4(_m1$2),this}rotateX(angle){return _m1$2.makeRotationX(angle),this.applyMatrix4(_m1$2),this}rotateY(angle){return _m1$2.makeRotationY(angle),this.applyMatrix4(_m1$2),this}rotateZ(angle){return _m1$2.makeRotationZ(angle),this.applyMatrix4(_m1$2),this}translate(x,y,z){return _m1$2.makeTranslation(x,y,z),this.applyMatrix4(_m1$2),this}scale(x,y,z){return _m1$2.makeScale(x,y,z),this.applyMatrix4(_m1$2),this}lookAt(vector){return _obj.lookAt(vector),_obj.updateMatrix(),this.applyMatrix4(_obj.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(_offset).negate(),this.translate(_offset.x,_offset.y,_offset.z),this}setFromPoints(points){const positionAttribute=this.getAttribute("position");if(positionAttribute===void 0){const position=[];for(let i=0,l=points.length;i<l;i++){const point=points[i];position.push(point.x,point.y,point.z||0)}this.setAttribute("position",new Float32BufferAttribute(position,3))}else{for(let i=0,l=positionAttribute.count;i<l;i++){const point=points[i];positionAttribute.setXYZ(i,point.x,point.y,point.z||0)}points.length>positionAttribute.count&&console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),positionAttribute.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Box3);const position=this.attributes.position,morphAttributesPosition=this.morphAttributes.position;if(position&&position.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new Vector3(-1/0,-1/0,-1/0),new Vector3(1/0,1/0,1/0));return}if(position!==void 0){if(this.boundingBox.setFromBufferAttribute(position),morphAttributesPosition)for(let i=0,il=morphAttributesPosition.length;i<il;i++){const morphAttribute=morphAttributesPosition[i];_box$2.setFromBufferAttribute(morphAttribute),this.morphTargetsRelative?(_vector$8.addVectors(this.boundingBox.min,_box$2.min),this.boundingBox.expandByPoint(_vector$8),_vector$8.addVectors(this.boundingBox.max,_box$2.max),this.boundingBox.expandByPoint(_vector$8)):(this.boundingBox.expandByPoint(_box$2.min),this.boundingBox.expandByPoint(_box$2.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Sphere);const position=this.attributes.position,morphAttributesPosition=this.morphAttributes.position;if(position&&position.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new Vector3,1/0);return}if(position){const center=this.boundingSphere.center;if(_box$2.setFromBufferAttribute(position),morphAttributesPosition)for(let i=0,il=morphAttributesPosition.length;i<il;i++){const morphAttribute=morphAttributesPosition[i];_boxMorphTargets.setFromBufferAttribute(morphAttribute),this.morphTargetsRelative?(_vector$8.addVectors(_box$2.min,_boxMorphTargets.min),_box$2.expandByPoint(_vector$8),_vector$8.addVectors(_box$2.max,_boxMorphTargets.max),_box$2.expandByPoint(_vector$8)):(_box$2.expandByPoint(_boxMorphTargets.min),_box$2.expandByPoint(_boxMorphTargets.max))}_box$2.getCenter(center);let maxRadiusSq=0;for(let i=0,il=position.count;i<il;i++)_vector$8.fromBufferAttribute(position,i),maxRadiusSq=Math.max(maxRadiusSq,center.distanceToSquared(_vector$8));if(morphAttributesPosition)for(let i=0,il=morphAttributesPosition.length;i<il;i++){const morphAttribute=morphAttributesPosition[i],morphTargetsRelative=this.morphTargetsRelative;for(let j=0,jl=morphAttribute.count;j<jl;j++)_vector$8.fromBufferAttribute(morphAttribute,j),morphTargetsRelative&&(_offset.fromBufferAttribute(position,j),_vector$8.add(_offset)),maxRadiusSq=Math.max(maxRadiusSq,center.distanceToSquared(_vector$8))}this.boundingSphere.radius=Math.sqrt(maxRadiusSq),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const index=this.index,attributes=this.attributes;if(index===null||attributes.position===void 0||attributes.normal===void 0||attributes.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const positionAttribute=attributes.position,normalAttribute=attributes.normal,uvAttribute=attributes.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new BufferAttribute(new Float32Array(4*positionAttribute.count),4));const tangentAttribute=this.getAttribute("tangent"),tan1=[],tan2=[];for(let i=0;i<positionAttribute.count;i++)tan1[i]=new Vector3,tan2[i]=new Vector3;const vA=new Vector3,vB=new Vector3,vC=new Vector3,uvA=new Vector2,uvB=new Vector2,uvC=new Vector2,sdir=new Vector3,tdir=new Vector3;function handleTriangle(a,b,c){vA.fromBufferAttribute(positionAttribute,a),vB.fromBufferAttribute(positionAttribute,b),vC.fromBufferAttribute(positionAttribute,c),uvA.fromBufferAttribute(uvAttribute,a),uvB.fromBufferAttribute(uvAttribute,b),uvC.fromBufferAttribute(uvAttribute,c),vB.sub(vA),vC.sub(vA),uvB.sub(uvA),uvC.sub(uvA);const r=1/(uvB.x*uvC.y-uvC.x*uvB.y);isFinite(r)&&(sdir.copy(vB).multiplyScalar(uvC.y).addScaledVector(vC,-uvB.y).multiplyScalar(r),tdir.copy(vC).multiplyScalar(uvB.x).addScaledVector(vB,-uvC.x).multiplyScalar(r),tan1[a].add(sdir),tan1[b].add(sdir),tan1[c].add(sdir),tan2[a].add(tdir),tan2[b].add(tdir),tan2[c].add(tdir))}__name(handleTriangle,"handleTriangle");let groups=this.groups;groups.length===0&&(groups=[{start:0,count:index.count}]);for(let i=0,il=groups.length;i<il;++i){const group=groups[i],start=group.start,count=group.count;for(let j=start,jl=start+count;j<jl;j+=3)handleTriangle(index.getX(j+0),index.getX(j+1),index.getX(j+2))}const tmp=new Vector3,tmp2=new Vector3,n=new Vector3,n2=new Vector3;function handleVertex(v){n.fromBufferAttribute(normalAttribute,v),n2.copy(n);const t2=tan1[v];tmp.copy(t2),tmp.sub(n.multiplyScalar(n.dot(t2))).normalize(),tmp2.crossVectors(n2,t2);const w=tmp2.dot(tan2[v])<0?-1:1;tangentAttribute.setXYZW(v,tmp.x,tmp.y,tmp.z,w)}__name(handleVertex,"handleVertex");for(let i=0,il=groups.length;i<il;++i){const group=groups[i],start=group.start,count=group.count;for(let j=start,jl=start+count;j<jl;j+=3)handleVertex(index.getX(j+0)),handleVertex(index.getX(j+1)),handleVertex(index.getX(j+2))}}computeVertexNormals(){const index=this.index,positionAttribute=this.getAttribute("position");if(positionAttribute!==void 0){let normalAttribute=this.getAttribute("normal");if(normalAttribute===void 0)normalAttribute=new BufferAttribute(new Float32Array(positionAttribute.count*3),3),this.setAttribute("normal",normalAttribute);else for(let i=0,il=normalAttribute.count;i<il;i++)normalAttribute.setXYZ(i,0,0,0);const pA=new Vector3,pB=new Vector3,pC=new Vector3,nA=new Vector3,nB=new Vector3,nC=new Vector3,cb=new Vector3,ab=new Vector3;if(index)for(let i=0,il=index.count;i<il;i+=3){const vA=index.getX(i+0),vB=index.getX(i+1),vC=index.getX(i+2);pA.fromBufferAttribute(positionAttribute,vA),pB.fromBufferAttribute(positionAttribute,vB),pC.fromBufferAttribute(positionAttribute,vC),cb.subVectors(pC,pB),ab.subVectors(pA,pB),cb.cross(ab),nA.fromBufferAttribute(normalAttribute,vA),nB.fromBufferAttribute(normalAttribute,vB),nC.fromBufferAttribute(normalAttribute,vC),nA.add(cb),nB.add(cb),nC.add(cb),normalAttribute.setXYZ(vA,nA.x,nA.y,nA.z),normalAttribute.setXYZ(vB,nB.x,nB.y,nB.z),normalAttribute.setXYZ(vC,nC.x,nC.y,nC.z)}else for(let i=0,il=positionAttribute.count;i<il;i+=3)pA.fromBufferAttribute(positionAttribute,i+0),pB.fromBufferAttribute(positionAttribute,i+1),pC.fromBufferAttribute(positionAttribute,i+2),cb.subVectors(pC,pB),ab.subVectors(pA,pB),cb.cross(ab),normalAttribute.setXYZ(i+0,cb.x,cb.y,cb.z),normalAttribute.setXYZ(i+1,cb.x,cb.y,cb.z),normalAttribute.setXYZ(i+2,cb.x,cb.y,cb.z);this.normalizeNormals(),normalAttribute.needsUpdate=!0}}normalizeNormals(){const normals=this.attributes.normal;for(let i=0,il=normals.count;i<il;i++)_vector$8.fromBufferAttribute(normals,i),_vector$8.normalize(),normals.setXYZ(i,_vector$8.x,_vector$8.y,_vector$8.z)}toNonIndexed(){function convertBufferAttribute(attribute,indices2){const array=attribute.array,itemSize=attribute.itemSize,normalized=attribute.normalized,array2=new array.constructor(indices2.length*itemSize);let index=0,index2=0;for(let i=0,l=indices2.length;i<l;i++){attribute.isInterleavedBufferAttribute?index=indices2[i]*attribute.data.stride+attribute.offset:index=indices2[i]*itemSize;for(let j=0;j<itemSize;j++)array2[index2++]=array[index++]}return new BufferAttribute(array2,itemSize,normalized)}if(__name(convertBufferAttribute,"convertBufferAttribute"),this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const geometry2=new BufferGeometry,indices=this.index.array,attributes=this.attributes;for(const name in attributes){const attribute=attributes[name],newAttribute=convertBufferAttribute(attribute,indices);geometry2.setAttribute(name,newAttribute)}const morphAttributes=this.morphAttributes;for(const name in morphAttributes){const morphArray=[],morphAttribute=morphAttributes[name];for(let i=0,il=morphAttribute.length;i<il;i++){const attribute=morphAttribute[i],newAttribute=convertBufferAttribute(attribute,indices);morphArray.push(newAttribute)}geometry2.morphAttributes[name]=morphArray}geometry2.morphTargetsRelative=this.morphTargetsRelative;const groups=this.groups;for(let i=0,l=groups.length;i<l;i++){const group=groups[i];geometry2.addGroup(group.start,group.count,group.materialIndex)}return geometry2}toJSON(){const data={metadata:{version:4.6,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(data.uuid=this.uuid,data.type=this.type,this.name!==""&&(data.name=this.name),Object.keys(this.userData).length>0&&(data.userData=this.userData),this.parameters!==void 0){const parameters=this.parameters;for(const key in parameters)parameters[key]!==void 0&&(data[key]=parameters[key]);return data}data.data={attributes:{}};const index=this.index;index!==null&&(data.data.index={type:index.array.constructor.name,array:Array.prototype.slice.call(index.array)});const attributes=this.attributes;for(const key in attributes){const attribute=attributes[key];data.data.attributes[key]=attribute.toJSON(data.data)}const morphAttributes={};let hasMorphAttributes=!1;for(const key in this.morphAttributes){const attributeArray=this.morphAttributes[key],array=[];for(let i=0,il=attributeArray.length;i<il;i++){const attribute=attributeArray[i];array.push(attribute.toJSON(data.data))}array.length>0&&(morphAttributes[key]=array,hasMorphAttributes=!0)}hasMorphAttributes&&(data.data.morphAttributes=morphAttributes,data.data.morphTargetsRelative=this.morphTargetsRelative);const groups=this.groups;groups.length>0&&(data.data.groups=JSON.parse(JSON.stringify(groups)));const boundingSphere=this.boundingSphere;return boundingSphere!==null&&(data.data.boundingSphere={center:boundingSphere.center.toArray(),radius:boundingSphere.radius}),data}clone(){return new this.constructor().copy(this)}copy(source){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const data={};this.name=source.name;const index=source.index;index!==null&&this.setIndex(index.clone(data));const attributes=source.attributes;for(const name in attributes){const attribute=attributes[name];this.setAttribute(name,attribute.clone(data))}const morphAttributes=source.morphAttributes;for(const name in morphAttributes){const array=[],morphAttribute=morphAttributes[name];for(let i=0,l=morphAttribute.length;i<l;i++)array.push(morphAttribute[i].clone(data));this.morphAttributes[name]=array}this.morphTargetsRelative=source.morphTargetsRelative;const groups=source.groups;for(let i=0,l=groups.length;i<l;i++){const group=groups[i];this.addGroup(group.start,group.count,group.materialIndex)}const boundingBox=source.boundingBox;boundingBox!==null&&(this.boundingBox=boundingBox.clone());const boundingSphere=source.boundingSphere;return boundingSphere!==null&&(this.boundingSphere=boundingSphere.clone()),this.drawRange.start=source.drawRange.start,this.drawRange.count=source.drawRange.count,this.userData=source.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const _inverseMatrix$3=new Matrix4,_ray$3=new Ray,_sphere$6=new Sphere,_sphereHitAt=new Vector3,_vA$1=new Vector3,_vB$1=new Vector3,_vC$1=new Vector3,_tempA=new Vector3,_morphA=new Vector3,_intersectionPoint=new Vector3,_intersectionPointWorld=new Vector3;class Mesh extends Object3D{static{__name(this,"Mesh")}constructor(geometry=new BufferGeometry,material=new MeshBasicMaterial){super(),this.isMesh=!0,this.type="Mesh",this.geometry=geometry,this.material=material,this.updateMorphTargets()}copy(source,recursive){return super.copy(source,recursive),source.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=source.morphTargetInfluences.slice()),source.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},source.morphTargetDictionary)),this.material=Array.isArray(source.material)?source.material.slice():source.material,this.geometry=source.geometry,this}updateMorphTargets(){const morphAttributes=this.geometry.morphAttributes,keys=Object.keys(morphAttributes);if(keys.length>0){const morphAttribute=morphAttributes[keys[0]];if(morphAttribute!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let m=0,ml=morphAttribute.length;m<ml;m++){const name=morphAttribute[m].name||String(m);this.morphTargetInfluences.push(0),this.morphTargetDictionary[name]=m}}}}getVertexPosition(index,target){const geometry=this.geometry,position=geometry.attributes.position,morphPosition=geometry.morphAttributes.position,morphTargetsRelative=geometry.morphTargetsRelative;target.fromBufferAttribute(position,index);const morphInfluences=this.morphTargetInfluences;if(morphPosition&&morphInfluences){_morphA.set(0,0,0);for(let i=0,il=morphPosition.length;i<il;i++){const influence=morphInfluences[i],morphAttribute=morphPosition[i];influence!==0&&(_tempA.fromBufferAttribute(morphAttribute,index),morphTargetsRelative?_morphA.addScaledVector(_tempA,influence):_morphA.addScaledVector(_tempA.sub(target),influence))}target.add(_morphA)}return target}raycast(raycaster,intersects2){const geometry=this.geometry,material=this.material,matrixWorld=this.matrixWorld;material!==void 0&&(geometry.boundingSphere===null&&geometry.computeBoundingSphere(),_sphere$6.copy(geometry.boundingSphere),_sphere$6.applyMatrix4(matrixWorld),_ray$3.copy(raycaster.ray).recast(raycaster.near),!(_sphere$6.containsPoint(_ray$3.origin)===!1&&(_ray$3.intersectSphere(_sphere$6,_sphereHitAt)===null||_ray$3.origin.distanceToSquared(_sphereHitAt)>(raycaster.far-raycaster.near)**2))&&(_inverseMatrix$3.copy(matrixWorld).invert(),_ray$3.copy(raycaster.ray).applyMatrix4(_inverseMatrix$3),!(geometry.boundingBox!==null&&_ray$3.intersectsBox(geometry.boundingBox)===!1)&&this._computeIntersections(raycaster,intersects2,_ray$3)))}_computeIntersections(raycaster,intersects2,rayLocalSpace){let intersection;const geometry=this.geometry,material=this.material,index=geometry.index,position=geometry.attributes.position,uv=geometry.attributes.uv,uv1=geometry.attributes.uv1,normal=geometry.attributes.normal,groups=geometry.groups,drawRange=geometry.drawRange;if(index!==null)if(Array.isArray(material))for(let i=0,il=groups.length;i<il;i++){const group=groups[i],groupMaterial=material[group.materialIndex],start=Math.max(group.start,drawRange.start),end=Math.min(index.count,Math.min(group.start+group.count,drawRange.start+drawRange.count));for(let j=start,jl=end;j<jl;j+=3){const a=index.getX(j),b=index.getX(j+1),c=index.getX(j+2);intersection=checkGeometryIntersection(this,groupMaterial,raycaster,rayLocalSpace,uv,uv1,normal,a,b,c),intersection&&(intersection.faceIndex=Math.floor(j/3),intersection.face.materialIndex=group.materialIndex,intersects2.push(intersection))}}else{const start=Math.max(0,drawRange.start),end=Math.min(index.count,drawRange.start+drawRange.count);for(let i=start,il=end;i<il;i+=3){const a=index.getX(i),b=index.getX(i+1),c=index.getX(i+2);intersection=checkGeometryIntersection(this,material,raycaster,rayLocalSpace,uv,uv1,normal,a,b,c),intersection&&(intersection.faceIndex=Math.floor(i/3),intersects2.push(intersection))}}else if(position!==void 0)if(Array.isArray(material))for(let i=0,il=groups.length;i<il;i++){const group=groups[i],groupMaterial=material[group.materialIndex],start=Math.max(group.start,drawRange.start),end=Math.min(position.count,Math.min(group.start+group.count,drawRange.start+drawRange.count));for(let j=start,jl=end;j<jl;j+=3){const a=j,b=j+1,c=j+2;intersection=checkGeometryIntersection(this,groupMaterial,raycaster,rayLocalSpace,uv,uv1,normal,a,b,c),intersection&&(intersection.faceIndex=Math.floor(j/3),intersection.face.materialIndex=group.materialIndex,intersects2.push(intersection))}}else{const start=Math.max(0,drawRange.start),end=Math.min(position.count,drawRange.start+drawRange.count);for(let i=start,il=end;i<il;i+=3){const a=i,b=i+1,c=i+2;intersection=checkGeometryIntersection(this,material,raycaster,rayLocalSpace,uv,uv1,normal,a,b,c),intersection&&(intersection.faceIndex=Math.floor(i/3),intersects2.push(intersection))}}}}function checkIntersection$1(object,material,raycaster,ray,pA,pB,pC,point){let intersect;if(material.side===BackSide?intersect=ray.intersectTriangle(pC,pB,pA,!0,point):intersect=ray.intersectTriangle(pA,pB,pC,material.side===FrontSide,point),intersect===null)return null;_intersectionPointWorld.copy(point),_intersectionPointWorld.applyMatrix4(object.matrixWorld);const distance=raycaster.ray.origin.distanceTo(_intersectionPointWorld);return distance<raycaster.near||distance>raycaster.far?null:{distance,point:_intersectionPointWorld.clone(),object}}__name(checkIntersection$1,"checkIntersection$1");function checkGeometryIntersection(object,material,raycaster,ray,uv,uv1,normal,a,b,c){object.getVertexPosition(a,_vA$1),object.getVertexPosition(b,_vB$1),object.getVertexPosition(c,_vC$1);const intersection=checkIntersection$1(object,material,raycaster,ray,_vA$1,_vB$1,_vC$1,_intersectionPoint);if(intersection){const barycoord=new Vector3;Triangle.getBarycoord(_intersectionPoint,_vA$1,_vB$1,_vC$1,barycoord),uv&&(intersection.uv=Triangle.getInterpolatedAttribute(uv,a,b,c,barycoord,new Vector2)),uv1&&(intersection.uv1=Triangle.getInterpolatedAttribute(uv1,a,b,c,barycoord,new Vector2)),normal&&(intersection.normal=Triangle.getInterpolatedAttribute(normal,a,b,c,barycoord,new Vector3),intersection.normal.dot(ray.direction)>0&&intersection.normal.multiplyScalar(-1));const face={a,b,c,normal:new Vector3,materialIndex:0};Triangle.getNormal(_vA$1,_vB$1,_vC$1,face.normal),intersection.face=face,intersection.barycoord=barycoord}return intersection}__name(checkGeometryIntersection,"checkGeometryIntersection");class BoxGeometry extends BufferGeometry{static{__name(this,"BoxGeometry")}constructor(width=1,height=1,depth=1,widthSegments=1,heightSegments=1,depthSegments=1){super(),this.type="BoxGeometry",this.parameters={width,height,depth,widthSegments,heightSegments,depthSegments};const scope=this;widthSegments=Math.floor(widthSegments),heightSegments=Math.floor(heightSegments),depthSegments=Math.floor(depthSegments);const indices=[],vertices=[],normals=[],uvs=[];let numberOfVertices=0,groupStart=0;buildPlane("z","y","x",-1,-1,depth,height,width,depthSegments,heightSegments,0),buildPlane("z","y","x",1,-1,depth,height,-width,depthSegments,heightSegments,1),buildPlane("x","z","y",1,1,width,depth,height,widthSegments,depthSegments,2),buildPlane("x","z","y",1,-1,width,depth,-height,widthSegments,depthSegments,3),buildPlane("x","y","z",1,-1,width,height,depth,widthSegments,heightSegments,4),buildPlane("x","y","z",-1,-1,width,height,-depth,widthSegments,heightSegments,5),this.setIndex(indices),this.setAttribute("position",new Float32BufferAttribute(vertices,3)),this.setAttribute("normal",new Float32BufferAttribute(normals,3)),this.setAttribute("uv",new Float32BufferAttribute(uvs,2));function buildPlane(u,v,w,udir,vdir,width2,height2,depth2,gridX,gridY,materialIndex){const segmentWidth=width2/gridX,segmentHeight=height2/gridY,widthHalf=width2/2,heightHalf=height2/2,depthHalf=depth2/2,gridX1=gridX+1,gridY1=gridY+1;let vertexCounter=0,groupCount=0;const vector=new Vector3;for(let iy=0;iy<gridY1;iy++){const y=iy*segmentHeight-heightHalf;for(let ix=0;ix<gridX1;ix++){const x=ix*segmentWidth-widthHalf;vector[u]=x*udir,vector[v]=y*vdir,vector[w]=depthHalf,vertices.push(vector.x,vector.y,vector.z),vector[u]=0,vector[v]=0,vector[w]=depth2>0?1:-1,normals.push(vector.x,vector.y,vector.z),uvs.push(ix/gridX),uvs.push(1-iy/gridY),vertexCounter+=1}}for(let iy=0;iy<gridY;iy++)for(let ix=0;ix<gridX;ix++){const a=numberOfVertices+ix+gridX1*iy,b=numberOfVertices+ix+gridX1*(iy+1),c=numberOfVertices+(ix+1)+gridX1*(iy+1),d=numberOfVertices+(ix+1)+gridX1*iy;indices.push(a,b,d),indices.push(b,c,d),groupCount+=6}scope.addGroup(groupStart,groupCount,materialIndex),groupStart+=groupCount,numberOfVertices+=vertexCounter}__name(buildPlane,"buildPlane")}copy(source){return super.copy(source),this.parameters=Object.assign({},source.parameters),this}static fromJSON(data){return new BoxGeometry(data.width,data.height,data.depth,data.widthSegments,data.heightSegments,data.depthSegments)}}function cloneUniforms(src){const dst={};for(const u in src){dst[u]={};for(const p in src[u]){const property=src[u][p];property&&(property.isColor||property.isMatrix3||property.isMatrix4||property.isVector2||property.isVector3||property.isVector4||property.isTexture||property.isQuaternion)?property.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),dst[u][p]=null):dst[u][p]=property.clone():Array.isArray(property)?dst[u][p]=property.slice():dst[u][p]=property}}return dst}__name(cloneUniforms,"cloneUniforms");function mergeUniforms(uniforms){const merged={};for(let u=0;u<uniforms.length;u++){const tmp=cloneUniforms(uniforms[u]);for(const p in tmp)merged[p]=tmp[p]}return merged}__name(mergeUniforms,"mergeUniforms");function cloneUniformsGroups(src){const dst=[];for(let u=0;u<src.length;u++)dst.push(src[u].clone());return dst}__name(cloneUniformsGroups,"cloneUniformsGroups");function getUnlitUniformColorSpace(renderer){const currentRenderTarget=renderer.getRenderTarget();return currentRenderTarget===null?renderer.outputColorSpace:currentRenderTarget.isXRRenderTarget===!0?currentRenderTarget.texture.colorSpace:ColorManagement.workingColorSpace}__name(getUnlitUniformColorSpace,"getUnlitUniformColorSpace");const UniformsUtils={clone:cloneUniforms,merge:mergeUniforms};var default_vertex=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,default_fragment=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class ShaderMaterial extends Material{static{__name(this,"ShaderMaterial")}static get type(){return"ShaderMaterial"}constructor(parameters){super(),this.isShaderMaterial=!0,this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=default_vertex,this.fragmentShader=default_fragment,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,parameters!==void 0&&this.setValues(parameters)}copy(source){return super.copy(source),this.fragmentShader=source.fragmentShader,this.vertexShader=source.vertexShader,this.uniforms=cloneUniforms(source.uniforms),this.uniformsGroups=cloneUniformsGroups(source.uniformsGroups),this.defines=Object.assign({},source.defines),this.wireframe=source.wireframe,this.wireframeLinewidth=source.wireframeLinewidth,this.fog=source.fog,this.lights=source.lights,this.clipping=source.clipping,this.extensions=Object.assign({},source.extensions),this.glslVersion=source.glslVersion,this}toJSON(meta){const data=super.toJSON(meta);data.glslVersion=this.glslVersion,data.uniforms={};for(const name in this.uniforms){const value=this.uniforms[name].value;value&&value.isTexture?data.uniforms[name]={type:"t",value:value.toJSON(meta).uuid}:value&&value.isColor?data.uniforms[name]={type:"c",value:value.getHex()}:value&&value.isVector2?data.uniforms[name]={type:"v2",value:value.toArray()}:value&&value.isVector3?data.uniforms[name]={type:"v3",value:value.toArray()}:value&&value.isVector4?data.uniforms[name]={type:"v4",value:value.toArray()}:value&&value.isMatrix3?data.uniforms[name]={type:"m3",value:value.toArray()}:value&&value.isMatrix4?data.uniforms[name]={type:"m4",value:value.toArray()}:data.uniforms[name]={value}}Object.keys(this.defines).length>0&&(data.defines=this.defines),data.vertexShader=this.vertexShader,data.fragmentShader=this.fragmentShader,data.lights=this.lights,data.clipping=this.clipping;const extensions={};for(const key in this.extensions)this.extensions[key]===!0&&(extensions[key]=!0);return Object.keys(extensions).length>0&&(data.extensions=extensions),data}}class Camera extends Object3D{static{__name(this,"Camera")}constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Matrix4,this.projectionMatrix=new Matrix4,this.projectionMatrixInverse=new Matrix4,this.coordinateSystem=WebGLCoordinateSystem}copy(source,recursive){return super.copy(source,recursive),this.matrixWorldInverse.copy(source.matrixWorldInverse),this.projectionMatrix.copy(source.projectionMatrix),this.projectionMatrixInverse.copy(source.projectionMatrixInverse),this.coordinateSystem=source.coordinateSystem,this}getWorldDirection(target){return super.getWorldDirection(target).negate()}updateMatrixWorld(force){super.updateMatrixWorld(force),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(updateParents,updateChildren){super.updateWorldMatrix(updateParents,updateChildren),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const _v3$1=new Vector3,_minTarget=new Vector2,_maxTarget=new Vector2;class PerspectiveCamera extends Camera{static{__name(this,"PerspectiveCamera")}constructor(fov2=50,aspect2=1,near=.1,far=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=fov2,this.zoom=1,this.near=near,this.far=far,this.focus=10,this.aspect=aspect2,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(source,recursive){return super.copy(source,recursive),this.fov=source.fov,this.zoom=source.zoom,this.near=source.near,this.far=source.far,this.focus=source.focus,this.aspect=source.aspect,this.view=source.view===null?null:Object.assign({},source.view),this.filmGauge=source.filmGauge,this.filmOffset=source.filmOffset,this}setFocalLength(focalLength){const vExtentSlope=.5*this.getFilmHeight()/focalLength;this.fov=RAD2DEG*2*Math.atan(vExtentSlope),this.updateProjectionMatrix()}getFocalLength(){const vExtentSlope=Math.tan(DEG2RAD*.5*this.fov);return .5*this.getFilmHeight()/vExtentSlope}getEffectiveFOV(){return RAD2DEG*2*Math.atan(Math.tan(DEG2RAD*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(distance,minTarget,maxTarget){_v3$1.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),minTarget.set(_v3$1.x,_v3$1.y).multiplyScalar(-distance/_v3$1.z),_v3$1.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),maxTarget.set(_v3$1.x,_v3$1.y).multiplyScalar(-distance/_v3$1.z)}getViewSize(distance,target){return this.getViewBounds(distance,_minTarget,_maxTarget),target.subVectors(_maxTarget,_minTarget)}setViewOffset(fullWidth,fullHeight,x,y,width,height){this.aspect=fullWidth/fullHeight,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=fullWidth,this.view.fullHeight=fullHeight,this.view.offsetX=x,this.view.offsetY=y,this.view.width=width,this.view.height=height,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const near=this.near;let top=near*Math.tan(DEG2RAD*.5*this.fov)/this.zoom,height=2*top,width=this.aspect*height,left=-.5*width;const view=this.view;if(this.view!==null&&this.view.enabled){const fullWidth=view.fullWidth,fullHeight=view.fullHeight;left+=view.offsetX*width/fullWidth,top-=view.offsetY*height/fullHeight,width*=view.width/fullWidth,height*=view.height/fullHeight}const skew=this.filmOffset;skew!==0&&(left+=near*skew/this.getFilmWidth()),this.projectionMatrix.makePerspective(left,left+width,top,top-height,near,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(meta){const data=super.toJSON(meta);return data.object.fov=this.fov,data.object.zoom=this.zoom,data.object.near=this.near,data.object.far=this.far,data.object.focus=this.focus,data.object.aspect=this.aspect,this.view!==null&&(data.object.view=Object.assign({},this.view)),data.object.filmGauge=this.filmGauge,data.object.filmOffset=this.filmOffset,data}}const fov=-90,aspect=1;class CubeCamera extends Object3D{static{__name(this,"CubeCamera")}constructor(near,far,renderTarget){super(),this.type="CubeCamera",this.renderTarget=renderTarget,this.coordinateSystem=null,this.activeMipmapLevel=0;const cameraPX=new PerspectiveCamera(fov,aspect,near,far);cameraPX.layers=this.layers,this.add(cameraPX);const cameraNX=new PerspectiveCamera(fov,aspect,near,far);cameraNX.layers=this.layers,this.add(cameraNX);const cameraPY=new PerspectiveCamera(fov,aspect,near,far);cameraPY.layers=this.layers,this.add(cameraPY);const cameraNY=new PerspectiveCamera(fov,aspect,near,far);cameraNY.layers=this.layers,this.add(cameraNY);const cameraPZ=new PerspectiveCamera(fov,aspect,near,far);cameraPZ.layers=this.layers,this.add(cameraPZ);const cameraNZ=new PerspectiveCamera(fov,aspect,near,far);cameraNZ.layers=this.layers,this.add(cameraNZ)}updateCoordinateSystem(){const coordinateSystem=this.coordinateSystem,cameras=this.children.concat(),[cameraPX,cameraNX,cameraPY,cameraNY,cameraPZ,cameraNZ]=cameras;for(const camera of cameras)this.remove(camera);if(coordinateSystem===WebGLCoordinateSystem)cameraPX.up.set(0,1,0),cameraPX.lookAt(1,0,0),cameraNX.up.set(0,1,0),cameraNX.lookAt(-1,0,0),cameraPY.up.set(0,0,-1),cameraPY.lookAt(0,1,0),cameraNY.up.set(0,0,1),cameraNY.lookAt(0,-1,0),cameraPZ.up.set(0,1,0),cameraPZ.lookAt(0,0,1),cameraNZ.up.set(0,1,0),cameraNZ.lookAt(0,0,-1);else if(coordinateSystem===WebGPUCoordinateSystem)cameraPX.up.set(0,-1,0),cameraPX.lookAt(-1,0,0),cameraNX.up.set(0,-1,0),cameraNX.lookAt(1,0,0),cameraPY.up.set(0,0,1),cameraPY.lookAt(0,1,0),cameraNY.up.set(0,0,-1),cameraNY.lookAt(0,-1,0),cameraPZ.up.set(0,-1,0),cameraPZ.lookAt(0,0,1),cameraNZ.up.set(0,-1,0),cameraNZ.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+coordinateSystem);for(const camera of cameras)this.add(camera),camera.updateMatrixWorld()}update(renderer,scene){this.parent===null&&this.updateMatrixWorld();const{renderTarget,activeMipmapLevel}=this;this.coordinateSystem!==renderer.coordinateSystem&&(this.coordinateSystem=renderer.coordinateSystem,this.updateCoordinateSystem());const[cameraPX,cameraNX,cameraPY,cameraNY,cameraPZ,cameraNZ]=this.children,currentRenderTarget=renderer.getRenderTarget(),currentActiveCubeFace=renderer.getActiveCubeFace(),currentActiveMipmapLevel=renderer.getActiveMipmapLevel(),currentXrEnabled=renderer.xr.enabled;renderer.xr.enabled=!1;const generateMipmaps=renderTarget.texture.generateMipmaps;renderTarget.texture.generateMipmaps=!1,renderer.setRenderTarget(renderTarget,0,activeMipmapLevel),renderer.render(scene,cameraPX),renderer.setRenderTarget(renderTarget,1,activeMipmapLevel),renderer.render(scene,cameraNX),renderer.setRenderTarget(renderTarget,2,activeMipmapLevel),renderer.render(scene,cameraPY),renderer.setRenderTarget(renderTarget,3,activeMipmapLevel),renderer.render(scene,cameraNY),renderer.setRenderTarget(renderTarget,4,activeMipmapLevel),renderer.render(scene,cameraPZ),renderTarget.texture.generateMipmaps=generateMipmaps,renderer.setRenderTarget(renderTarget,5,activeMipmapLevel),renderer.render(scene,cameraNZ),renderer.setRenderTarget(currentRenderTarget,currentActiveCubeFace,currentActiveMipmapLevel),renderer.xr.enabled=currentXrEnabled,renderTarget.texture.needsPMREMUpdate=!0}}class CubeTexture extends Texture{static{__name(this,"CubeTexture")}constructor(images,mapping,wrapS,wrapT,magFilter,minFilter,format,type,anisotropy,colorSpace){images=images!==void 0?images:[],mapping=mapping!==void 0?mapping:CubeReflectionMapping,super(images,mapping,wrapS,wrapT,magFilter,minFilter,format,type,anisotropy,colorSpace),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(value){this.image=value}}class WebGLCubeRenderTarget extends WebGLRenderTarget{static{__name(this,"WebGLCubeRenderTarget")}constructor(size=1,options={}){super(size,size,options),this.isWebGLCubeRenderTarget=!0;const image={width:size,height:size,depth:1},images=[image,image,image,image,image,image];this.texture=new CubeTexture(images,options.mapping,options.wrapS,options.wrapT,options.magFilter,options.minFilter,options.format,options.type,options.anisotropy,options.colorSpace),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=options.generateMipmaps!==void 0?options.generateMipmaps:!1,this.texture.minFilter=options.minFilter!==void 0?options.minFilter:LinearFilter}fromEquirectangularTexture(renderer,texture){this.texture.type=texture.type,this.texture.colorSpace=texture.colorSpace,this.texture.generateMipmaps=texture.generateMipmaps,this.texture.minFilter=texture.minFilter,this.texture.magFilter=texture.magFilter;const shader={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},geometry=new BoxGeometry(5,5,5),material=new ShaderMaterial({name:"CubemapFromEquirect",uniforms:cloneUniforms(shader.uniforms),vertexShader:shader.vertexShader,fragmentShader:shader.fragmentShader,side:BackSide,blending:NoBlending});material.uniforms.tEquirect.value=texture;const mesh=new Mesh(geometry,material),currentMinFilter=texture.minFilter;return texture.minFilter===LinearMipmapLinearFilter&&(texture.minFilter=LinearFilter),new CubeCamera(1,10,this).update(renderer,mesh),texture.minFilter=currentMinFilter,mesh.geometry.dispose(),mesh.material.dispose(),this}clear(renderer,color,depth,stencil){const currentRenderTarget=renderer.getRenderTarget();for(let i=0;i<6;i++)renderer.setRenderTarget(this,i),renderer.clear(color,depth,stencil);renderer.setRenderTarget(currentRenderTarget)}}const _vector1=new Vector3,_vector2=new Vector3,_normalMatrix=new Matrix3;class Plane{static{__name(this,"Plane")}constructor(normal=new Vector3(1,0,0),constant=0){this.isPlane=!0,this.normal=normal,this.constant=constant}set(normal,constant){return this.normal.copy(normal),this.constant=constant,this}setComponents(x,y,z,w){return this.normal.set(x,y,z),this.constant=w,this}setFromNormalAndCoplanarPoint(normal,point){return this.normal.copy(normal),this.constant=-point.dot(this.normal),this}setFromCoplanarPoints(a,b,c){const normal=_vector1.subVectors(c,b).cross(_vector2.subVectors(a,b)).normalize();return this.setFromNormalAndCoplanarPoint(normal,a),this}copy(plane){return this.normal.copy(plane.normal),this.constant=plane.constant,this}normalize(){const inverseNormalLength=1/this.normal.length();return this.normal.multiplyScalar(inverseNormalLength),this.constant*=inverseNormalLength,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(point){return this.normal.dot(point)+this.constant}distanceToSphere(sphere){return this.distanceToPoint(sphere.center)-sphere.radius}projectPoint(point,target){return target.copy(point).addScaledVector(this.normal,-this.distanceToPoint(point))}intersectLine(line,target){const direction=line.delta(_vector1),denominator=this.normal.dot(direction);if(denominator===0)return this.distanceToPoint(line.start)===0?target.copy(line.start):null;const t2=-(line.start.dot(this.normal)+this.constant)/denominator;return t2<0||t2>1?null:target.copy(line.start).addScaledVector(direction,t2)}intersectsLine(line){const startSign=this.distanceToPoint(line.start),endSign=this.distanceToPoint(line.end);return startSign<0&&endSign>0||endSign<0&&startSign>0}intersectsBox(box){return box.intersectsPlane(this)}intersectsSphere(sphere){return sphere.intersectsPlane(this)}coplanarPoint(target){return target.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(matrix,optionalNormalMatrix){const normalMatrix=optionalNormalMatrix||_normalMatrix.getNormalMatrix(matrix),referencePoint=this.coplanarPoint(_vector1).applyMatrix4(matrix),normal=this.normal.applyMatrix3(normalMatrix).normalize();return this.constant=-referencePoint.dot(normal),this}translate(offset){return this.constant-=offset.dot(this.normal),this}equals(plane){return plane.normal.equals(this.normal)&&plane.constant===this.constant}clone(){return new this.constructor().copy(this)}}const _sphere$5=new Sphere,_vector$7=new Vector3;class Frustum{static{__name(this,"Frustum")}constructor(p0=new Plane,p1=new Plane,p2=new Plane,p3=new Plane,p4=new Plane,p5=new Plane){this.planes=[p0,p1,p2,p3,p4,p5]}set(p0,p1,p2,p3,p4,p5){const planes=this.planes;return planes[0].copy(p0),planes[1].copy(p1),planes[2].copy(p2),planes[3].copy(p3),planes[4].copy(p4),planes[5].copy(p5),this}copy(frustum){const planes=this.planes;for(let i=0;i<6;i++)planes[i].copy(frustum.planes[i]);return this}setFromProjectionMatrix(m,coordinateSystem=WebGLCoordinateSystem){const planes=this.planes,me=m.elements,me0=me[0],me1=me[1],me2=me[2],me3=me[3],me4=me[4],me5=me[5],me6=me[6],me7=me[7],me8=me[8],me9=me[9],me10=me[10],me11=me[11],me12=me[12],me13=me[13],me14=me[14],me15=me[15];if(planes[0].setComponents(me3-me0,me7-me4,me11-me8,me15-me12).normalize(),planes[1].setComponents(me3+me0,me7+me4,me11+me8,me15+me12).normalize(),planes[2].setComponents(me3+me1,me7+me5,me11+me9,me15+me13).normalize(),planes[3].setComponents(me3-me1,me7-me5,me11-me9,me15-me13).normalize(),planes[4].setComponents(me3-me2,me7-me6,me11-me10,me15-me14).normalize(),coordinateSystem===WebGLCoordinateSystem)planes[5].setComponents(me3+me2,me7+me6,me11+me10,me15+me14).normalize();else if(coordinateSystem===WebGPUCoordinateSystem)planes[5].setComponents(me2,me6,me10,me14).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+coordinateSystem);return this}intersectsObject(object){if(object.boundingSphere!==void 0)object.boundingSphere===null&&object.computeBoundingSphere(),_sphere$5.copy(object.boundingSphere).applyMatrix4(object.matrixWorld);else{const geometry=object.geometry;geometry.boundingSphere===null&&geometry.computeBoundingSphere(),_sphere$5.copy(geometry.boundingSphere).applyMatrix4(object.matrixWorld)}return this.intersectsSphere(_sphere$5)}intersectsSprite(sprite){return _sphere$5.center.set(0,0,0),_sphere$5.radius=.7071067811865476,_sphere$5.applyMatrix4(sprite.matrixWorld),this.intersectsSphere(_sphere$5)}intersectsSphere(sphere){const planes=this.planes,center=sphere.center,negRadius=-sphere.radius;for(let i=0;i<6;i++)if(planes[i].distanceToPoint(center)<negRadius)return!1;return!0}intersectsBox(box){const planes=this.planes;for(let i=0;i<6;i++){const plane=planes[i];if(_vector$7.x=plane.normal.x>0?box.max.x:box.min.x,_vector$7.y=plane.normal.y>0?box.max.y:box.min.y,_vector$7.z=plane.normal.z>0?box.max.z:box.min.z,plane.distanceToPoint(_vector$7)<0)return!1}return!0}containsPoint(point){const planes=this.planes;for(let i=0;i<6;i++)if(planes[i].distanceToPoint(point)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}function WebGLAnimation(){let context=null,isAnimating=!1,animationLoop=null,requestId=null;function onAnimationFrame(time,frame){animationLoop(time,frame),requestId=context.requestAnimationFrame(onAnimationFrame)}return __name(onAnimationFrame,"onAnimationFrame"),{start:__name(function(){isAnimating!==!0&&animationLoop!==null&&(requestId=context.requestAnimationFrame(onAnimationFrame),isAnimating=!0)},"start"),stop:__name(function(){context.cancelAnimationFrame(requestId),isAnimating=!1},"stop"),setAnimationLoop:__name(function(callback){animationLoop=callback},"setAnimationLoop"),setContext:__name(function(value){context=value},"setContext")}}__name(WebGLAnimation,"WebGLAnimation");function WebGLAttributes(gl){const buffers=new WeakMap;function createBuffer(attribute,bufferType){const array=attribute.array,usage=attribute.usage,size=array.byteLength,buffer=gl.createBuffer();gl.bindBuffer(bufferType,buffer),gl.bufferData(bufferType,array,usage),attribute.onUploadCallback();let type;if(array instanceof Float32Array)type=gl.FLOAT;else if(array instanceof Uint16Array)attribute.isFloat16BufferAttribute?type=gl.HALF_FLOAT:type=gl.UNSIGNED_SHORT;else if(array instanceof Int16Array)type=gl.SHORT;else if(array instanceof Uint32Array)type=gl.UNSIGNED_INT;else if(array instanceof Int32Array)type=gl.INT;else if(array instanceof Int8Array)type=gl.BYTE;else if(array instanceof Uint8Array)type=gl.UNSIGNED_BYTE;else if(array instanceof Uint8ClampedArray)type=gl.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+array);return{buffer,type,bytesPerElement:array.BYTES_PER_ELEMENT,version:attribute.version,size}}__name(createBuffer,"createBuffer");function updateBuffer(buffer,attribute,bufferType){const array=attribute.array,updateRanges=attribute.updateRanges;if(gl.bindBuffer(bufferType,buffer),updateRanges.length===0)gl.bufferSubData(bufferType,0,array);else{updateRanges.sort((a,b)=>a.start-b.start);let mergeIndex=0;for(let i=1;i<updateRanges.length;i++){const previousRange=updateRanges[mergeIndex],range=updateRanges[i];range.start<=previousRange.start+previousRange.count+1?previousRange.count=Math.max(previousRange.count,range.start+range.count-previousRange.start):(++mergeIndex,updateRanges[mergeIndex]=range)}updateRanges.length=mergeIndex+1;for(let i=0,l=updateRanges.length;i<l;i++){const range=updateRanges[i];gl.bufferSubData(bufferType,range.start*array.BYTES_PER_ELEMENT,array,range.start,range.count)}attribute.clearUpdateRanges()}attribute.onUploadCallback()}__name(updateBuffer,"updateBuffer");function get(attribute){return attribute.isInterleavedBufferAttribute&&(attribute=attribute.data),buffers.get(attribute)}__name(get,"get");function remove(attribute){attribute.isInterleavedBufferAttribute&&(attribute=attribute.data);const data=buffers.get(attribute);data&&(gl.deleteBuffer(data.buffer),buffers.delete(attribute))}__name(remove,"remove");function update(attribute,bufferType){if(attribute.isInterleavedBufferAttribute&&(attribute=attribute.data),attribute.isGLBufferAttribute){const cached=buffers.get(attribute);(!cached||cached.version<attribute.version)&&buffers.set(attribute,{buffer:attribute.buffer,type:attribute.type,bytesPerElement:attribute.elementSize,version:attribute.version});return}const data=buffers.get(attribute);if(data===void 0)buffers.set(attribute,createBuffer(attribute,bufferType));else if(data.version<attribute.version){if(data.size!==attribute.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");updateBuffer(data.buffer,attribute,bufferType),data.version=attribute.version}}return __name(update,"update"),{get,remove,update}}__name(WebGLAttributes,"WebGLAttributes");class PlaneGeometry extends BufferGeometry{static{__name(this,"PlaneGeometry")}constructor(width=1,height=1,widthSegments=1,heightSegments=1){super(),this.type="PlaneGeometry",this.parameters={width,height,widthSegments,heightSegments};const width_half=width/2,height_half=height/2,gridX=Math.floor(widthSegments),gridY=Math.floor(heightSegments),gridX1=gridX+1,gridY1=gridY+1,segment_width=width/gridX,segment_height=height/gridY,indices=[],vertices=[],normals=[],uvs=[];for(let iy=0;iy<gridY1;iy++){const y=iy*segment_height-height_half;for(let ix=0;ix<gridX1;ix++){const x=ix*segment_width-width_half;vertices.push(x,-y,0),normals.push(0,0,1),uvs.push(ix/gridX),uvs.push(1-iy/gridY)}}for(let iy=0;iy<gridY;iy++)for(let ix=0;ix<gridX;ix++){const a=ix+gridX1*iy,b=ix+gridX1*(iy+1),c=ix+1+gridX1*(iy+1),d=ix+1+gridX1*iy;indices.push(a,b,d),indices.push(b,c,d)}this.setIndex(indices),this.setAttribute("position",new Float32BufferAttribute(vertices,3)),this.setAttribute("normal",new Float32BufferAttribute(normals,3)),this.setAttribute("uv",new Float32BufferAttribute(uvs,2))}copy(source){return super.copy(source),this.parameters=Object.assign({},source.parameters),this}static fromJSON(data){return new PlaneGeometry(data.width,data.height,data.widthSegments,data.heightSegments)}}var alphahash_fragment=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,alphahash_pars_fragment=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,alphamap_fragment=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,alphamap_pars_fragment=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,alphatest_fragment=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,alphatest_pars_fragment=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,aomap_fragment=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,aomap_pars_fragment=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,batching_pars_vertex=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,batching_vertex=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,begin_vertex=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,beginnormal_vertex=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,bsdfs=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,iridescence_fragment=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,bumpmap_pars_fragment=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,clipping_planes_fragment=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,clipping_planes_pars_fragment=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,clipping_planes_pars_vertex=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,clipping_planes_vertex=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,color_fragment=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,color_pars_fragment=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,color_pars_vertex=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,color_vertex=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,common=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,cube_uv_reflection_fragment=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,defaultnormal_vertex=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,displacementmap_pars_vertex=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,displacementmap_vertex=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,emissivemap_fragment=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,emissivemap_pars_fragment=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,colorspace_fragment="gl_FragColor = linearToOutputTexel( gl_FragColor );",colorspace_pars_fragment=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,envmap_fragment=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,envmap_common_pars_fragment=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,envmap_pars_fragment=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,envmap_pars_vertex=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,envmap_vertex=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,fog_vertex=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,fog_pars_vertex=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,fog_fragment=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,fog_pars_fragment=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,gradientmap_pars_fragment=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,lightmap_pars_fragment=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,lights_lambert_fragment=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,lights_lambert_pars_fragment=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,lights_pars_begin=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,envmap_physical_pars_fragment=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,lights_toon_fragment=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,lights_toon_pars_fragment=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,lights_phong_fragment=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,lights_phong_pars_fragment=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,lights_physical_fragment=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,lights_physical_pars_fragment=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,lights_fragment_begin=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,lights_fragment_maps=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,lights_fragment_end=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,logdepthbuf_fragment=`#if defined( USE_LOGDEPTHBUF )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,logdepthbuf_pars_fragment=`#if defined( USE_LOGDEPTHBUF )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,logdepthbuf_pars_vertex=`#ifdef USE_LOGDEPTHBUF
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,logdepthbuf_vertex=`#ifdef USE_LOGDEPTHBUF
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,map_fragment=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,map_pars_fragment=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,map_particle_fragment=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,map_particle_pars_fragment=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,metalnessmap_fragment=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,metalnessmap_pars_fragment=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,morphinstance_vertex=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,morphcolor_vertex=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,morphnormal_vertex=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,morphtarget_pars_vertex=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,morphtarget_vertex=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,normal_fragment_begin=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,normal_fragment_maps=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,normal_pars_fragment=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,normal_pars_vertex=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,normal_vertex=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,normalmap_pars_fragment=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,clearcoat_normal_fragment_begin=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,clearcoat_normal_fragment_maps=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,clearcoat_pars_fragment=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,iridescence_pars_fragment=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,opaque_fragment=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,packing=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,premultiplied_alpha_fragment=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,project_vertex=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,dithering_fragment=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,dithering_pars_fragment=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,roughnessmap_fragment=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,roughnessmap_pars_fragment=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,shadowmap_pars_fragment=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,shadowmap_pars_vertex=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,shadowmap_vertex=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,shadowmask_pars_fragment=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,skinbase_vertex=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,skinning_pars_vertex=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,skinning_vertex=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,skinnormal_vertex=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,specularmap_fragment=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,specularmap_pars_fragment=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,tonemapping_fragment=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,tonemapping_pars_fragment=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,transmission_fragment=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,transmission_pars_fragment=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
		
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
		
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		
		#else
		
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,uv_pars_fragment=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,uv_pars_vertex=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,uv_vertex=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,worldpos_vertex=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const vertex$h=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,fragment$h=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,vertex$g=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,fragment$g=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,vertex$f=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,fragment$f=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,vertex$e=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,fragment$e=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,vertex$d=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,fragment$d=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,vertex$c=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,fragment$c=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,vertex$b=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,fragment$b=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,vertex$a=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,fragment$a=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,vertex$9=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,fragment$9=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,vertex$8=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,fragment$8=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,vertex$7=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,fragment$7=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,vertex$6=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,fragment$6=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,vertex$5=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,fragment$5=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,vertex$4=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,fragment$4=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,vertex$3=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,fragment$3=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,vertex$2=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,fragment$2=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,vertex$1=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,fragment$1=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ShaderChunk={alphahash_fragment,alphahash_pars_fragment,alphamap_fragment,alphamap_pars_fragment,alphatest_fragment,alphatest_pars_fragment,aomap_fragment,aomap_pars_fragment,batching_pars_vertex,batching_vertex,begin_vertex,beginnormal_vertex,bsdfs,iridescence_fragment,bumpmap_pars_fragment,clipping_planes_fragment,clipping_planes_pars_fragment,clipping_planes_pars_vertex,clipping_planes_vertex,color_fragment,color_pars_fragment,color_pars_vertex,color_vertex,common,cube_uv_reflection_fragment,defaultnormal_vertex,displacementmap_pars_vertex,displacementmap_vertex,emissivemap_fragment,emissivemap_pars_fragment,colorspace_fragment,colorspace_pars_fragment,envmap_fragment,envmap_common_pars_fragment,envmap_pars_fragment,envmap_pars_vertex,envmap_physical_pars_fragment,envmap_vertex,fog_vertex,fog_pars_vertex,fog_fragment,fog_pars_fragment,gradientmap_pars_fragment,lightmap_pars_fragment,lights_lambert_fragment,lights_lambert_pars_fragment,lights_pars_begin,lights_toon_fragment,lights_toon_pars_fragment,lights_phong_fragment,lights_phong_pars_fragment,lights_physical_fragment,lights_physical_pars_fragment,lights_fragment_begin,lights_fragment_maps,lights_fragment_end,logdepthbuf_fragment,logdepthbuf_pars_fragment,logdepthbuf_pars_vertex,logdepthbuf_vertex,map_fragment,map_pars_fragment,map_particle_fragment,map_particle_pars_fragment,metalnessmap_fragment,metalnessmap_pars_fragment,morphinstance_vertex,morphcolor_vertex,morphnormal_vertex,morphtarget_pars_vertex,morphtarget_vertex,normal_fragment_begin,normal_fragment_maps,normal_pars_fragment,normal_pars_vertex,normal_vertex,normalmap_pars_fragment,clearcoat_normal_fragment_begin,clearcoat_normal_fragment_maps,clearcoat_pars_fragment,iridescence_pars_fragment,opaque_fragment,packing,premultiplied_alpha_fragment,project_vertex,dithering_fragment,dithering_pars_fragment,roughnessmap_fragment,roughnessmap_pars_fragment,shadowmap_pars_fragment,shadowmap_pars_vertex,shadowmap_vertex,shadowmask_pars_fragment,skinbase_vertex,skinning_pars_vertex,skinning_vertex,skinnormal_vertex,specularmap_fragment,specularmap_pars_fragment,tonemapping_fragment,tonemapping_pars_fragment,transmission_fragment,transmission_pars_fragment,uv_pars_fragment,uv_pars_vertex,uv_vertex,worldpos_vertex,background_vert:vertex$h,background_frag:fragment$h,backgroundCube_vert:vertex$g,backgroundCube_frag:fragment$g,cube_vert:vertex$f,cube_frag:fragment$f,depth_vert:vertex$e,depth_frag:fragment$e,distanceRGBA_vert:vertex$d,distanceRGBA_frag:fragment$d,equirect_vert:vertex$c,equirect_frag:fragment$c,linedashed_vert:vertex$b,linedashed_frag:fragment$b,meshbasic_vert:vertex$a,meshbasic_frag:fragment$a,meshlambert_vert:vertex$9,meshlambert_frag:fragment$9,meshmatcap_vert:vertex$8,meshmatcap_frag:fragment$8,meshnormal_vert:vertex$7,meshnormal_frag:fragment$7,meshphong_vert:vertex$6,meshphong_frag:fragment$6,meshphysical_vert:vertex$5,meshphysical_frag:fragment$5,meshtoon_vert:vertex$4,meshtoon_frag:fragment$4,points_vert:vertex$3,points_frag:fragment$3,shadow_vert:vertex$2,shadow_frag:fragment$2,sprite_vert:vertex$1,sprite_frag:fragment$1},UniformsLib={common:{diffuse:{value:new Color(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Matrix3},alphaMap:{value:null},alphaMapTransform:{value:new Matrix3},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Matrix3}},envmap:{envMap:{value:null},envMapRotation:{value:new Matrix3},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Matrix3}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Matrix3}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Matrix3},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Matrix3},normalScale:{value:new Vector2(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Matrix3},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Matrix3}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Matrix3}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Matrix3}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Color(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new Color(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Matrix3},alphaTest:{value:0},uvTransform:{value:new Matrix3}},sprite:{diffuse:{value:new Color(16777215)},opacity:{value:1},center:{value:new Vector2(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Matrix3},alphaMap:{value:null},alphaMapTransform:{value:new Matrix3},alphaTest:{value:0}}},ShaderLib={basic:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.specularmap,UniformsLib.envmap,UniformsLib.aomap,UniformsLib.lightmap,UniformsLib.fog]),vertexShader:ShaderChunk.meshbasic_vert,fragmentShader:ShaderChunk.meshbasic_frag},lambert:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.specularmap,UniformsLib.envmap,UniformsLib.aomap,UniformsLib.lightmap,UniformsLib.emissivemap,UniformsLib.bumpmap,UniformsLib.normalmap,UniformsLib.displacementmap,UniformsLib.fog,UniformsLib.lights,{emissive:{value:new Color(0)}}]),vertexShader:ShaderChunk.meshlambert_vert,fragmentShader:ShaderChunk.meshlambert_frag},phong:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.specularmap,UniformsLib.envmap,UniformsLib.aomap,UniformsLib.lightmap,UniformsLib.emissivemap,UniformsLib.bumpmap,UniformsLib.normalmap,UniformsLib.displacementmap,UniformsLib.fog,UniformsLib.lights,{emissive:{value:new Color(0)},specular:{value:new Color(1118481)},shininess:{value:30}}]),vertexShader:ShaderChunk.meshphong_vert,fragmentShader:ShaderChunk.meshphong_frag},standard:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.envmap,UniformsLib.aomap,UniformsLib.lightmap,UniformsLib.emissivemap,UniformsLib.bumpmap,UniformsLib.normalmap,UniformsLib.displacementmap,UniformsLib.roughnessmap,UniformsLib.metalnessmap,UniformsLib.fog,UniformsLib.lights,{emissive:{value:new Color(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:ShaderChunk.meshphysical_vert,fragmentShader:ShaderChunk.meshphysical_frag},toon:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.aomap,UniformsLib.lightmap,UniformsLib.emissivemap,UniformsLib.bumpmap,UniformsLib.normalmap,UniformsLib.displacementmap,UniformsLib.gradientmap,UniformsLib.fog,UniformsLib.lights,{emissive:{value:new Color(0)}}]),vertexShader:ShaderChunk.meshtoon_vert,fragmentShader:ShaderChunk.meshtoon_frag},matcap:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.bumpmap,UniformsLib.normalmap,UniformsLib.displacementmap,UniformsLib.fog,{matcap:{value:null}}]),vertexShader:ShaderChunk.meshmatcap_vert,fragmentShader:ShaderChunk.meshmatcap_frag},points:{uniforms:mergeUniforms([UniformsLib.points,UniformsLib.fog]),vertexShader:ShaderChunk.points_vert,fragmentShader:ShaderChunk.points_frag},dashed:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:ShaderChunk.linedashed_vert,fragmentShader:ShaderChunk.linedashed_frag},depth:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.displacementmap]),vertexShader:ShaderChunk.depth_vert,fragmentShader:ShaderChunk.depth_frag},normal:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.bumpmap,UniformsLib.normalmap,UniformsLib.displacementmap,{opacity:{value:1}}]),vertexShader:ShaderChunk.meshnormal_vert,fragmentShader:ShaderChunk.meshnormal_frag},sprite:{uniforms:mergeUniforms([UniformsLib.sprite,UniformsLib.fog]),vertexShader:ShaderChunk.sprite_vert,fragmentShader:ShaderChunk.sprite_frag},background:{uniforms:{uvTransform:{value:new Matrix3},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:ShaderChunk.background_vert,fragmentShader:ShaderChunk.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Matrix3}},vertexShader:ShaderChunk.backgroundCube_vert,fragmentShader:ShaderChunk.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:ShaderChunk.cube_vert,fragmentShader:ShaderChunk.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:ShaderChunk.equirect_vert,fragmentShader:ShaderChunk.equirect_frag},distanceRGBA:{uniforms:mergeUniforms([UniformsLib.common,UniformsLib.displacementmap,{referencePosition:{value:new Vector3},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:ShaderChunk.distanceRGBA_vert,fragmentShader:ShaderChunk.distanceRGBA_frag},shadow:{uniforms:mergeUniforms([UniformsLib.lights,UniformsLib.fog,{color:{value:new Color(0)},opacity:{value:1}}]),vertexShader:ShaderChunk.shadow_vert,fragmentShader:ShaderChunk.shadow_frag}};ShaderLib.physical={uniforms:mergeUniforms([ShaderLib.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Matrix3},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Matrix3},clearcoatNormalScale:{value:new Vector2(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Matrix3},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Matrix3},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Matrix3},sheen:{value:0},sheenColor:{value:new Color(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Matrix3},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Matrix3},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Matrix3},transmissionSamplerSize:{value:new Vector2},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Matrix3},attenuationDistance:{value:0},attenuationColor:{value:new Color(0)},specularColor:{value:new Color(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Matrix3},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Matrix3},anisotropyVector:{value:new Vector2},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Matrix3}}]),vertexShader:ShaderChunk.meshphysical_vert,fragmentShader:ShaderChunk.meshphysical_frag};const _rgb={r:0,b:0,g:0},_e1$1=new Euler,_m1$1=new Matrix4;function WebGLBackground(renderer,cubemaps,cubeuvmaps,state,objects,alpha,premultipliedAlpha){const clearColor=new Color(0);let clearAlpha=alpha===!0?0:1,planeMesh,boxMesh,currentBackground=null,currentBackgroundVersion=0,currentTonemapping=null;function getBackground(scene){let background=scene.isScene===!0?scene.background:null;return background&&background.isTexture&&(background=(scene.backgroundBlurriness>0?cubeuvmaps:cubemaps).get(background)),background}__name(getBackground,"getBackground");function render(scene){let forceClear=!1;const background=getBackground(scene);background===null?setClear(clearColor,clearAlpha):background&&background.isColor&&(setClear(background,1),forceClear=!0);const environmentBlendMode=renderer.xr.getEnvironmentBlendMode();environmentBlendMode==="additive"?state.buffers.color.setClear(0,0,0,1,premultipliedAlpha):environmentBlendMode==="alpha-blend"&&state.buffers.color.setClear(0,0,0,0,premultipliedAlpha),(renderer.autoClear||forceClear)&&(state.buffers.depth.setTest(!0),state.buffers.depth.setMask(!0),state.buffers.color.setMask(!0),renderer.clear(renderer.autoClearColor,renderer.autoClearDepth,renderer.autoClearStencil))}__name(render,"render");function addToRenderList(renderList,scene){const background=getBackground(scene);background&&(background.isCubeTexture||background.mapping===CubeUVReflectionMapping)?(boxMesh===void 0&&(boxMesh=new Mesh(new BoxGeometry(1,1,1),new ShaderMaterial({name:"BackgroundCubeMaterial",uniforms:cloneUniforms(ShaderLib.backgroundCube.uniforms),vertexShader:ShaderLib.backgroundCube.vertexShader,fragmentShader:ShaderLib.backgroundCube.fragmentShader,side:BackSide,depthTest:!1,depthWrite:!1,fog:!1})),boxMesh.geometry.deleteAttribute("normal"),boxMesh.geometry.deleteAttribute("uv"),boxMesh.onBeforeRender=function(renderer2,scene2,camera){this.matrixWorld.copyPosition(camera.matrixWorld)},Object.defineProperty(boxMesh.material,"envMap",{get:__name(function(){return this.uniforms.envMap.value},"get")}),objects.update(boxMesh)),_e1$1.copy(scene.backgroundRotation),_e1$1.x*=-1,_e1$1.y*=-1,_e1$1.z*=-1,background.isCubeTexture&&background.isRenderTargetTexture===!1&&(_e1$1.y*=-1,_e1$1.z*=-1),boxMesh.material.uniforms.envMap.value=background,boxMesh.material.uniforms.flipEnvMap.value=background.isCubeTexture&&background.isRenderTargetTexture===!1?-1:1,boxMesh.material.uniforms.backgroundBlurriness.value=scene.backgroundBlurriness,boxMesh.material.uniforms.backgroundIntensity.value=scene.backgroundIntensity,boxMesh.material.uniforms.backgroundRotation.value.setFromMatrix4(_m1$1.makeRotationFromEuler(_e1$1)),boxMesh.material.toneMapped=ColorManagement.getTransfer(background.colorSpace)!==SRGBTransfer,(currentBackground!==background||currentBackgroundVersion!==background.version||currentTonemapping!==renderer.toneMapping)&&(boxMesh.material.needsUpdate=!0,currentBackground=background,currentBackgroundVersion=background.version,currentTonemapping=renderer.toneMapping),boxMesh.layers.enableAll(),renderList.unshift(boxMesh,boxMesh.geometry,boxMesh.material,0,0,null)):background&&background.isTexture&&(planeMesh===void 0&&(planeMesh=new Mesh(new PlaneGeometry(2,2),new ShaderMaterial({name:"BackgroundMaterial",uniforms:cloneUniforms(ShaderLib.background.uniforms),vertexShader:ShaderLib.background.vertexShader,fragmentShader:ShaderLib.background.fragmentShader,side:FrontSide,depthTest:!1,depthWrite:!1,fog:!1})),planeMesh.geometry.deleteAttribute("normal"),Object.defineProperty(planeMesh.material,"map",{get:__name(function(){return this.uniforms.t2D.value},"get")}),objects.update(planeMesh)),planeMesh.material.uniforms.t2D.value=background,planeMesh.material.uniforms.backgroundIntensity.value=scene.backgroundIntensity,planeMesh.material.toneMapped=ColorManagement.getTransfer(background.colorSpace)!==SRGBTransfer,background.matrixAutoUpdate===!0&&background.updateMatrix(),planeMesh.material.uniforms.uvTransform.value.copy(background.matrix),(currentBackground!==background||currentBackgroundVersion!==background.version||currentTonemapping!==renderer.toneMapping)&&(planeMesh.material.needsUpdate=!0,currentBackground=background,currentBackgroundVersion=background.version,currentTonemapping=renderer.toneMapping),planeMesh.layers.enableAll(),renderList.unshift(planeMesh,planeMesh.geometry,planeMesh.material,0,0,null))}__name(addToRenderList,"addToRenderList");function setClear(color,alpha2){color.getRGB(_rgb,getUnlitUniformColorSpace(renderer)),state.buffers.color.setClear(_rgb.r,_rgb.g,_rgb.b,alpha2,premultipliedAlpha)}return __name(setClear,"setClear"),{getClearColor:__name(function(){return clearColor},"getClearColor"),setClearColor:__name(function(color,alpha2=1){clearColor.set(color),clearAlpha=alpha2,setClear(clearColor,clearAlpha)},"setClearColor"),getClearAlpha:__name(function(){return clearAlpha},"getClearAlpha"),setClearAlpha:__name(function(alpha2){clearAlpha=alpha2,setClear(clearColor,clearAlpha)},"setClearAlpha"),render,addToRenderList}}__name(WebGLBackground,"WebGLBackground");function WebGLBindingStates(gl,attributes){const maxVertexAttributes=gl.getParameter(gl.MAX_VERTEX_ATTRIBS),bindingStates={},defaultState=createBindingState(null);let currentState=defaultState,forceUpdate=!1;function setup(object,material,program,geometry,index){let updateBuffers=!1;const state=getBindingState(geometry,program,material);currentState!==state&&(currentState=state,bindVertexArrayObject(currentState.object)),updateBuffers=needsUpdate(object,geometry,program,index),updateBuffers&&saveCache(object,geometry,program,index),index!==null&&attributes.update(index,gl.ELEMENT_ARRAY_BUFFER),(updateBuffers||forceUpdate)&&(forceUpdate=!1,setupVertexAttributes(object,material,program,geometry),index!==null&&gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,attributes.get(index).buffer))}__name(setup,"setup");function createVertexArrayObject(){return gl.createVertexArray()}__name(createVertexArrayObject,"createVertexArrayObject");function bindVertexArrayObject(vao){return gl.bindVertexArray(vao)}__name(bindVertexArrayObject,"bindVertexArrayObject");function deleteVertexArrayObject(vao){return gl.deleteVertexArray(vao)}__name(deleteVertexArrayObject,"deleteVertexArrayObject");function getBindingState(geometry,program,material){const wireframe=material.wireframe===!0;let programMap=bindingStates[geometry.id];programMap===void 0&&(programMap={},bindingStates[geometry.id]=programMap);let stateMap=programMap[program.id];stateMap===void 0&&(stateMap={},programMap[program.id]=stateMap);let state=stateMap[wireframe];return state===void 0&&(state=createBindingState(createVertexArrayObject()),stateMap[wireframe]=state),state}__name(getBindingState,"getBindingState");function createBindingState(vao){const newAttributes=[],enabledAttributes=[],attributeDivisors=[];for(let i=0;i<maxVertexAttributes;i++)newAttributes[i]=0,enabledAttributes[i]=0,attributeDivisors[i]=0;return{geometry:null,program:null,wireframe:!1,newAttributes,enabledAttributes,attributeDivisors,object:vao,attributes:{},index:null}}__name(createBindingState,"createBindingState");function needsUpdate(object,geometry,program,index){const cachedAttributes=currentState.attributes,geometryAttributes=geometry.attributes;let attributesNum=0;const programAttributes=program.getAttributes();for(const name in programAttributes)if(programAttributes[name].location>=0){const cachedAttribute=cachedAttributes[name];let geometryAttribute=geometryAttributes[name];if(geometryAttribute===void 0&&(name==="instanceMatrix"&&object.instanceMatrix&&(geometryAttribute=object.instanceMatrix),name==="instanceColor"&&object.instanceColor&&(geometryAttribute=object.instanceColor)),cachedAttribute===void 0||cachedAttribute.attribute!==geometryAttribute||geometryAttribute&&cachedAttribute.data!==geometryAttribute.data)return!0;attributesNum++}return currentState.attributesNum!==attributesNum||currentState.index!==index}__name(needsUpdate,"needsUpdate");function saveCache(object,geometry,program,index){const cache={},attributes2=geometry.attributes;let attributesNum=0;const programAttributes=program.getAttributes();for(const name in programAttributes)if(programAttributes[name].location>=0){let attribute=attributes2[name];attribute===void 0&&(name==="instanceMatrix"&&object.instanceMatrix&&(attribute=object.instanceMatrix),name==="instanceColor"&&object.instanceColor&&(attribute=object.instanceColor));const data={};data.attribute=attribute,attribute&&attribute.data&&(data.data=attribute.data),cache[name]=data,attributesNum++}currentState.attributes=cache,currentState.attributesNum=attributesNum,currentState.index=index}__name(saveCache,"saveCache");function initAttributes(){const newAttributes=currentState.newAttributes;for(let i=0,il=newAttributes.length;i<il;i++)newAttributes[i]=0}__name(initAttributes,"initAttributes");function enableAttribute(attribute){enableAttributeAndDivisor(attribute,0)}__name(enableAttribute,"enableAttribute");function enableAttributeAndDivisor(attribute,meshPerAttribute){const newAttributes=currentState.newAttributes,enabledAttributes=currentState.enabledAttributes,attributeDivisors=currentState.attributeDivisors;newAttributes[attribute]=1,enabledAttributes[attribute]===0&&(gl.enableVertexAttribArray(attribute),enabledAttributes[attribute]=1),attributeDivisors[attribute]!==meshPerAttribute&&(gl.vertexAttribDivisor(attribute,meshPerAttribute),attributeDivisors[attribute]=meshPerAttribute)}__name(enableAttributeAndDivisor,"enableAttributeAndDivisor");function disableUnusedAttributes(){const newAttributes=currentState.newAttributes,enabledAttributes=currentState.enabledAttributes;for(let i=0,il=enabledAttributes.length;i<il;i++)enabledAttributes[i]!==newAttributes[i]&&(gl.disableVertexAttribArray(i),enabledAttributes[i]=0)}__name(disableUnusedAttributes,"disableUnusedAttributes");function vertexAttribPointer(index,size,type,normalized,stride,offset,integer){integer===!0?gl.vertexAttribIPointer(index,size,type,stride,offset):gl.vertexAttribPointer(index,size,type,normalized,stride,offset)}__name(vertexAttribPointer,"vertexAttribPointer");function setupVertexAttributes(object,material,program,geometry){initAttributes();const geometryAttributes=geometry.attributes,programAttributes=program.getAttributes(),materialDefaultAttributeValues=material.defaultAttributeValues;for(const name in programAttributes){const programAttribute=programAttributes[name];if(programAttribute.location>=0){let geometryAttribute=geometryAttributes[name];if(geometryAttribute===void 0&&(name==="instanceMatrix"&&object.instanceMatrix&&(geometryAttribute=object.instanceMatrix),name==="instanceColor"&&object.instanceColor&&(geometryAttribute=object.instanceColor)),geometryAttribute!==void 0){const normalized=geometryAttribute.normalized,size=geometryAttribute.itemSize,attribute=attributes.get(geometryAttribute);if(attribute===void 0)continue;const buffer=attribute.buffer,type=attribute.type,bytesPerElement=attribute.bytesPerElement,integer=type===gl.INT||type===gl.UNSIGNED_INT||geometryAttribute.gpuType===IntType;if(geometryAttribute.isInterleavedBufferAttribute){const data=geometryAttribute.data,stride=data.stride,offset=geometryAttribute.offset;if(data.isInstancedInterleavedBuffer){for(let i=0;i<programAttribute.locationSize;i++)enableAttributeAndDivisor(programAttribute.location+i,data.meshPerAttribute);object.isInstancedMesh!==!0&&geometry._maxInstanceCount===void 0&&(geometry._maxInstanceCount=data.meshPerAttribute*data.count)}else for(let i=0;i<programAttribute.locationSize;i++)enableAttribute(programAttribute.location+i);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(let i=0;i<programAttribute.locationSize;i++)vertexAttribPointer(programAttribute.location+i,size/programAttribute.locationSize,type,normalized,stride*bytesPerElement,(offset+size/programAttribute.locationSize*i)*bytesPerElement,integer)}else{if(geometryAttribute.isInstancedBufferAttribute){for(let i=0;i<programAttribute.locationSize;i++)enableAttributeAndDivisor(programAttribute.location+i,geometryAttribute.meshPerAttribute);object.isInstancedMesh!==!0&&geometry._maxInstanceCount===void 0&&(geometry._maxInstanceCount=geometryAttribute.meshPerAttribute*geometryAttribute.count)}else for(let i=0;i<programAttribute.locationSize;i++)enableAttribute(programAttribute.location+i);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(let i=0;i<programAttribute.locationSize;i++)vertexAttribPointer(programAttribute.location+i,size/programAttribute.locationSize,type,normalized,size*bytesPerElement,size/programAttribute.locationSize*i*bytesPerElement,integer)}}else if(materialDefaultAttributeValues!==void 0){const value=materialDefaultAttributeValues[name];if(value!==void 0)switch(value.length){case 2:gl.vertexAttrib2fv(programAttribute.location,value);break;case 3:gl.vertexAttrib3fv(programAttribute.location,value);break;case 4:gl.vertexAttrib4fv(programAttribute.location,value);break;default:gl.vertexAttrib1fv(programAttribute.location,value)}}}}disableUnusedAttributes()}__name(setupVertexAttributes,"setupVertexAttributes");function dispose(){reset();for(const geometryId in bindingStates){const programMap=bindingStates[geometryId];for(const programId in programMap){const stateMap=programMap[programId];for(const wireframe in stateMap)deleteVertexArrayObject(stateMap[wireframe].object),delete stateMap[wireframe];delete programMap[programId]}delete bindingStates[geometryId]}}__name(dispose,"dispose");function releaseStatesOfGeometry(geometry){if(bindingStates[geometry.id]===void 0)return;const programMap=bindingStates[geometry.id];for(const programId in programMap){const stateMap=programMap[programId];for(const wireframe in stateMap)deleteVertexArrayObject(stateMap[wireframe].object),delete stateMap[wireframe];delete programMap[programId]}delete bindingStates[geometry.id]}__name(releaseStatesOfGeometry,"releaseStatesOfGeometry");function releaseStatesOfProgram(program){for(const geometryId in bindingStates){const programMap=bindingStates[geometryId];if(programMap[program.id]===void 0)continue;const stateMap=programMap[program.id];for(const wireframe in stateMap)deleteVertexArrayObject(stateMap[wireframe].object),delete stateMap[wireframe];delete programMap[program.id]}}__name(releaseStatesOfProgram,"releaseStatesOfProgram");function reset(){resetDefaultState(),forceUpdate=!0,currentState!==defaultState&&(currentState=defaultState,bindVertexArrayObject(currentState.object))}__name(reset,"reset");function resetDefaultState(){defaultState.geometry=null,defaultState.program=null,defaultState.wireframe=!1}return __name(resetDefaultState,"resetDefaultState"),{setup,reset,resetDefaultState,dispose,releaseStatesOfGeometry,releaseStatesOfProgram,initAttributes,enableAttribute,disableUnusedAttributes}}__name(WebGLBindingStates,"WebGLBindingStates");function WebGLBufferRenderer(gl,extensions,info){let mode;function setMode(value){mode=value}__name(setMode,"setMode");function render(start,count){gl.drawArrays(mode,start,count),info.update(count,mode,1)}__name(render,"render");function renderInstances(start,count,primcount){primcount!==0&&(gl.drawArraysInstanced(mode,start,count,primcount),info.update(count,mode,primcount))}__name(renderInstances,"renderInstances");function renderMultiDraw(starts,counts,drawCount){if(drawCount===0)return;extensions.get("WEBGL_multi_draw").multiDrawArraysWEBGL(mode,starts,0,counts,0,drawCount);let elementCount=0;for(let i=0;i<drawCount;i++)elementCount+=counts[i];info.update(elementCount,mode,1)}__name(renderMultiDraw,"renderMultiDraw");function renderMultiDrawInstances(starts,counts,drawCount,primcount){if(drawCount===0)return;const extension=extensions.get("WEBGL_multi_draw");if(extension===null)for(let i=0;i<starts.length;i++)renderInstances(starts[i],counts[i],primcount[i]);else{extension.multiDrawArraysInstancedWEBGL(mode,starts,0,counts,0,primcount,0,drawCount);let elementCount=0;for(let i=0;i<drawCount;i++)elementCount+=counts[i]*primcount[i];info.update(elementCount,mode,1)}}__name(renderMultiDrawInstances,"renderMultiDrawInstances"),this.setMode=setMode,this.render=render,this.renderInstances=renderInstances,this.renderMultiDraw=renderMultiDraw,this.renderMultiDrawInstances=renderMultiDrawInstances}__name(WebGLBufferRenderer,"WebGLBufferRenderer");function WebGLCapabilities(gl,extensions,parameters,utils){let maxAnisotropy;function getMaxAnisotropy(){if(maxAnisotropy!==void 0)return maxAnisotropy;if(extensions.has("EXT_texture_filter_anisotropic")===!0){const extension=extensions.get("EXT_texture_filter_anisotropic");maxAnisotropy=gl.getParameter(extension.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else maxAnisotropy=0;return maxAnisotropy}__name(getMaxAnisotropy,"getMaxAnisotropy");function textureFormatReadable(textureFormat){return!(textureFormat!==RGBAFormat&&utils.convert(textureFormat)!==gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT))}__name(textureFormatReadable,"textureFormatReadable");function textureTypeReadable(textureType){const halfFloatSupportedByExt=textureType===HalfFloatType&&(extensions.has("EXT_color_buffer_half_float")||extensions.has("EXT_color_buffer_float"));return!(textureType!==UnsignedByteType&&utils.convert(textureType)!==gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE)&&textureType!==FloatType&&!halfFloatSupportedByExt)}__name(textureTypeReadable,"textureTypeReadable");function getMaxPrecision(precision2){if(precision2==="highp"){if(gl.getShaderPrecisionFormat(gl.VERTEX_SHADER,gl.HIGH_FLOAT).precision>0&&gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER,gl.HIGH_FLOAT).precision>0)return"highp";precision2="mediump"}return precision2==="mediump"&&gl.getShaderPrecisionFormat(gl.VERTEX_SHADER,gl.MEDIUM_FLOAT).precision>0&&gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER,gl.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}__name(getMaxPrecision,"getMaxPrecision");let precision=parameters.precision!==void 0?parameters.precision:"highp";const maxPrecision=getMaxPrecision(precision);maxPrecision!==precision&&(console.warn("THREE.WebGLRenderer:",precision,"not supported, using",maxPrecision,"instead."),precision=maxPrecision);const logarithmicDepthBuffer=parameters.logarithmicDepthBuffer===!0,reverseDepthBuffer=parameters.reverseDepthBuffer===!0&&extensions.has("EXT_clip_control"),maxTextures=gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),maxVertexTextures=gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),maxTextureSize=gl.getParameter(gl.MAX_TEXTURE_SIZE),maxCubemapSize=gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE),maxAttributes=gl.getParameter(gl.MAX_VERTEX_ATTRIBS),maxVertexUniforms=gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),maxVaryings=gl.getParameter(gl.MAX_VARYING_VECTORS),maxFragmentUniforms=gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),vertexTextures=maxVertexTextures>0,maxSamples=gl.getParameter(gl.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy,getMaxPrecision,textureFormatReadable,textureTypeReadable,precision,logarithmicDepthBuffer,reverseDepthBuffer,maxTextures,maxVertexTextures,maxTextureSize,maxCubemapSize,maxAttributes,maxVertexUniforms,maxVaryings,maxFragmentUniforms,vertexTextures,maxSamples}}__name(WebGLCapabilities,"WebGLCapabilities");function WebGLClipping(properties){const scope=this;let globalState=null,numGlobalPlanes=0,localClippingEnabled=!1,renderingShadows=!1;const plane=new Plane,viewNormalMatrix=new Matrix3,uniform={value:null,needsUpdate:!1};this.uniform=uniform,this.numPlanes=0,this.numIntersection=0,this.init=function(planes,enableLocalClipping){const enabled=planes.length!==0||enableLocalClipping||numGlobalPlanes!==0||localClippingEnabled;return localClippingEnabled=enableLocalClipping,numGlobalPlanes=planes.length,enabled},this.beginShadows=function(){renderingShadows=!0,projectPlanes(null)},this.endShadows=function(){renderingShadows=!1},this.setGlobalState=function(planes,camera){globalState=projectPlanes(planes,camera,0)},this.setState=function(material,camera,useCache){const planes=material.clippingPlanes,clipIntersection=material.clipIntersection,clipShadows=material.clipShadows,materialProperties=properties.get(material);if(!localClippingEnabled||planes===null||planes.length===0||renderingShadows&&!clipShadows)renderingShadows?projectPlanes(null):resetGlobalState();else{const nGlobal=renderingShadows?0:numGlobalPlanes,lGlobal=nGlobal*4;let dstArray=materialProperties.clippingState||null;uniform.value=dstArray,dstArray=projectPlanes(planes,camera,lGlobal,useCache);for(let i=0;i!==lGlobal;++i)dstArray[i]=globalState[i];materialProperties.clippingState=dstArray,this.numIntersection=clipIntersection?this.numPlanes:0,this.numPlanes+=nGlobal}};function resetGlobalState(){uniform.value!==globalState&&(uniform.value=globalState,uniform.needsUpdate=numGlobalPlanes>0),scope.numPlanes=numGlobalPlanes,scope.numIntersection=0}__name(resetGlobalState,"resetGlobalState");function projectPlanes(planes,camera,dstOffset,skipTransform){const nPlanes=planes!==null?planes.length:0;let dstArray=null;if(nPlanes!==0){if(dstArray=uniform.value,skipTransform!==!0||dstArray===null){const flatSize=dstOffset+nPlanes*4,viewMatrix=camera.matrixWorldInverse;viewNormalMatrix.getNormalMatrix(viewMatrix),(dstArray===null||dstArray.length<flatSize)&&(dstArray=new Float32Array(flatSize));for(let i=0,i4=dstOffset;i!==nPlanes;++i,i4+=4)plane.copy(planes[i]).applyMatrix4(viewMatrix,viewNormalMatrix),plane.normal.toArray(dstArray,i4),dstArray[i4+3]=plane.constant}uniform.value=dstArray,uniform.needsUpdate=!0}return scope.numPlanes=nPlanes,scope.numIntersection=0,dstArray}__name(projectPlanes,"projectPlanes")}__name(WebGLClipping,"WebGLClipping");function WebGLCubeMaps(renderer){let cubemaps=new WeakMap;function mapTextureMapping(texture,mapping){return mapping===EquirectangularReflectionMapping?texture.mapping=CubeReflectionMapping:mapping===EquirectangularRefractionMapping&&(texture.mapping=CubeRefractionMapping),texture}__name(mapTextureMapping,"mapTextureMapping");function get(texture){if(texture&&texture.isTexture){const mapping=texture.mapping;if(mapping===EquirectangularReflectionMapping||mapping===EquirectangularRefractionMapping)if(cubemaps.has(texture)){const cubemap=cubemaps.get(texture).texture;return mapTextureMapping(cubemap,texture.mapping)}else{const image=texture.image;if(image&&image.height>0){const renderTarget=new WebGLCubeRenderTarget(image.height);return renderTarget.fromEquirectangularTexture(renderer,texture),cubemaps.set(texture,renderTarget),texture.addEventListener("dispose",onTextureDispose),mapTextureMapping(renderTarget.texture,texture.mapping)}else return null}}return texture}__name(get,"get");function onTextureDispose(event){const texture=event.target;texture.removeEventListener("dispose",onTextureDispose);const cubemap=cubemaps.get(texture);cubemap!==void 0&&(cubemaps.delete(texture),cubemap.dispose())}__name(onTextureDispose,"onTextureDispose");function dispose(){cubemaps=new WeakMap}return __name(dispose,"dispose"),{get,dispose}}__name(WebGLCubeMaps,"WebGLCubeMaps");class OrthographicCamera extends Camera{static{__name(this,"OrthographicCamera")}constructor(left=-1,right=1,top=1,bottom=-1,near=.1,far=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=left,this.right=right,this.top=top,this.bottom=bottom,this.near=near,this.far=far,this.updateProjectionMatrix()}copy(source,recursive){return super.copy(source,recursive),this.left=source.left,this.right=source.right,this.top=source.top,this.bottom=source.bottom,this.near=source.near,this.far=source.far,this.zoom=source.zoom,this.view=source.view===null?null:Object.assign({},source.view),this}setViewOffset(fullWidth,fullHeight,x,y,width,height){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=fullWidth,this.view.fullHeight=fullHeight,this.view.offsetX=x,this.view.offsetY=y,this.view.width=width,this.view.height=height,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const dx=(this.right-this.left)/(2*this.zoom),dy=(this.top-this.bottom)/(2*this.zoom),cx=(this.right+this.left)/2,cy=(this.top+this.bottom)/2;let left=cx-dx,right=cx+dx,top=cy+dy,bottom=cy-dy;if(this.view!==null&&this.view.enabled){const scaleW=(this.right-this.left)/this.view.fullWidth/this.zoom,scaleH=(this.top-this.bottom)/this.view.fullHeight/this.zoom;left+=scaleW*this.view.offsetX,right=left+scaleW*this.view.width,top-=scaleH*this.view.offsetY,bottom=top-scaleH*this.view.height}this.projectionMatrix.makeOrthographic(left,right,top,bottom,this.near,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(meta){const data=super.toJSON(meta);return data.object.zoom=this.zoom,data.object.left=this.left,data.object.right=this.right,data.object.top=this.top,data.object.bottom=this.bottom,data.object.near=this.near,data.object.far=this.far,this.view!==null&&(data.object.view=Object.assign({},this.view)),data}}const LOD_MIN=4,EXTRA_LOD_SIGMA=[.125,.215,.35,.446,.526,.582],MAX_SAMPLES=20,_flatCamera=new OrthographicCamera,_clearColor=new Color;let _oldTarget=null,_oldActiveCubeFace=0,_oldActiveMipmapLevel=0,_oldXrEnabled=!1;const PHI=(1+Math.sqrt(5))/2,INV_PHI=1/PHI,_axisDirections=[new Vector3(-PHI,INV_PHI,0),new Vector3(PHI,INV_PHI,0),new Vector3(-INV_PHI,0,PHI),new Vector3(INV_PHI,0,PHI),new Vector3(0,PHI,-INV_PHI),new Vector3(0,PHI,INV_PHI),new Vector3(-1,1,-1),new Vector3(1,1,-1),new Vector3(-1,1,1),new Vector3(1,1,1)];class PMREMGenerator{static{__name(this,"PMREMGenerator")}constructor(renderer){this._renderer=renderer,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(scene,sigma=0,near=.1,far=100){_oldTarget=this._renderer.getRenderTarget(),_oldActiveCubeFace=this._renderer.getActiveCubeFace(),_oldActiveMipmapLevel=this._renderer.getActiveMipmapLevel(),_oldXrEnabled=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(256);const cubeUVRenderTarget=this._allocateTargets();return cubeUVRenderTarget.depthBuffer=!0,this._sceneToCubeUV(scene,near,far,cubeUVRenderTarget),sigma>0&&this._blur(cubeUVRenderTarget,0,0,sigma),this._applyPMREM(cubeUVRenderTarget),this._cleanup(cubeUVRenderTarget),cubeUVRenderTarget}fromEquirectangular(equirectangular,renderTarget=null){return this._fromTexture(equirectangular,renderTarget)}fromCubemap(cubemap,renderTarget=null){return this._fromTexture(cubemap,renderTarget)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=_getCubemapMaterial(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=_getEquirectMaterial(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(cubeSize){this._lodMax=Math.floor(Math.log2(cubeSize)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let i=0;i<this._lodPlanes.length;i++)this._lodPlanes[i].dispose()}_cleanup(outputTarget){this._renderer.setRenderTarget(_oldTarget,_oldActiveCubeFace,_oldActiveMipmapLevel),this._renderer.xr.enabled=_oldXrEnabled,outputTarget.scissorTest=!1,_setViewport(outputTarget,0,0,outputTarget.width,outputTarget.height)}_fromTexture(texture,renderTarget){texture.mapping===CubeReflectionMapping||texture.mapping===CubeRefractionMapping?this._setSize(texture.image.length===0?16:texture.image[0].width||texture.image[0].image.width):this._setSize(texture.image.width/4),_oldTarget=this._renderer.getRenderTarget(),_oldActiveCubeFace=this._renderer.getActiveCubeFace(),_oldActiveMipmapLevel=this._renderer.getActiveMipmapLevel(),_oldXrEnabled=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const cubeUVRenderTarget=renderTarget||this._allocateTargets();return this._textureToCubeUV(texture,cubeUVRenderTarget),this._applyPMREM(cubeUVRenderTarget),this._cleanup(cubeUVRenderTarget),cubeUVRenderTarget}_allocateTargets(){const width=3*Math.max(this._cubeSize,112),height=4*this._cubeSize,params={magFilter:LinearFilter,minFilter:LinearFilter,generateMipmaps:!1,type:HalfFloatType,format:RGBAFormat,colorSpace:LinearSRGBColorSpace,depthBuffer:!1},cubeUVRenderTarget=_createRenderTarget(width,height,params);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==width||this._pingPongRenderTarget.height!==height){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=_createRenderTarget(width,height,params);const{_lodMax}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=_createPlanes(_lodMax)),this._blurMaterial=_getBlurShader(_lodMax,width,height)}return cubeUVRenderTarget}_compileMaterial(material){const tmpMesh=new Mesh(this._lodPlanes[0],material);this._renderer.compile(tmpMesh,_flatCamera)}_sceneToCubeUV(scene,near,far,cubeUVRenderTarget){const cubeCamera=new PerspectiveCamera(90,1,near,far),upSign=[1,-1,1,1,1,1],forwardSign=[1,1,1,-1,-1,-1],renderer=this._renderer,originalAutoClear=renderer.autoClear,toneMapping=renderer.toneMapping;renderer.getClearColor(_clearColor),renderer.toneMapping=NoToneMapping,renderer.autoClear=!1;const backgroundMaterial=new MeshBasicMaterial({name:"PMREM.Background",side:BackSide,depthWrite:!1,depthTest:!1}),backgroundBox=new Mesh(new BoxGeometry,backgroundMaterial);let useSolidColor=!1;const background=scene.background;background?background.isColor&&(backgroundMaterial.color.copy(background),scene.background=null,useSolidColor=!0):(backgroundMaterial.color.copy(_clearColor),useSolidColor=!0);for(let i=0;i<6;i++){const col=i%3;col===0?(cubeCamera.up.set(0,upSign[i],0),cubeCamera.lookAt(forwardSign[i],0,0)):col===1?(cubeCamera.up.set(0,0,upSign[i]),cubeCamera.lookAt(0,forwardSign[i],0)):(cubeCamera.up.set(0,upSign[i],0),cubeCamera.lookAt(0,0,forwardSign[i]));const size=this._cubeSize;_setViewport(cubeUVRenderTarget,col*size,i>2?size:0,size,size),renderer.setRenderTarget(cubeUVRenderTarget),useSolidColor&&renderer.render(backgroundBox,cubeCamera),renderer.render(scene,cubeCamera)}backgroundBox.geometry.dispose(),backgroundBox.material.dispose(),renderer.toneMapping=toneMapping,renderer.autoClear=originalAutoClear,scene.background=background}_textureToCubeUV(texture,cubeUVRenderTarget){const renderer=this._renderer,isCubeTexture=texture.mapping===CubeReflectionMapping||texture.mapping===CubeRefractionMapping;isCubeTexture?(this._cubemapMaterial===null&&(this._cubemapMaterial=_getCubemapMaterial()),this._cubemapMaterial.uniforms.flipEnvMap.value=texture.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=_getEquirectMaterial());const material=isCubeTexture?this._cubemapMaterial:this._equirectMaterial,mesh=new Mesh(this._lodPlanes[0],material),uniforms=material.uniforms;uniforms.envMap.value=texture;const size=this._cubeSize;_setViewport(cubeUVRenderTarget,0,0,3*size,2*size),renderer.setRenderTarget(cubeUVRenderTarget),renderer.render(mesh,_flatCamera)}_applyPMREM(cubeUVRenderTarget){const renderer=this._renderer,autoClear=renderer.autoClear;renderer.autoClear=!1;const n=this._lodPlanes.length;for(let i=1;i<n;i++){const sigma=Math.sqrt(this._sigmas[i]*this._sigmas[i]-this._sigmas[i-1]*this._sigmas[i-1]),poleAxis=_axisDirections[(n-i-1)%_axisDirections.length];this._blur(cubeUVRenderTarget,i-1,i,sigma,poleAxis)}renderer.autoClear=autoClear}_blur(cubeUVRenderTarget,lodIn,lodOut,sigma,poleAxis){const pingPongRenderTarget=this._pingPongRenderTarget;this._halfBlur(cubeUVRenderTarget,pingPongRenderTarget,lodIn,lodOut,sigma,"latitudinal",poleAxis),this._halfBlur(pingPongRenderTarget,cubeUVRenderTarget,lodOut,lodOut,sigma,"longitudinal",poleAxis)}_halfBlur(targetIn,targetOut,lodIn,lodOut,sigmaRadians,direction,poleAxis){const renderer=this._renderer,blurMaterial=this._blurMaterial;direction!=="latitudinal"&&direction!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");const STANDARD_DEVIATIONS=3,blurMesh=new Mesh(this._lodPlanes[lodOut],blurMaterial),blurUniforms=blurMaterial.uniforms,pixels=this._sizeLods[lodIn]-1,radiansPerPixel=isFinite(sigmaRadians)?Math.PI/(2*pixels):2*Math.PI/(2*MAX_SAMPLES-1),sigmaPixels=sigmaRadians/radiansPerPixel,samples=isFinite(sigmaRadians)?1+Math.floor(STANDARD_DEVIATIONS*sigmaPixels):MAX_SAMPLES;samples>MAX_SAMPLES&&console.warn(`sigmaRadians, ${sigmaRadians}, is too large and will clip, as it requested ${samples} samples when the maximum is set to ${MAX_SAMPLES}`);const weights=[];let sum=0;for(let i=0;i<MAX_SAMPLES;++i){const x2=i/sigmaPixels,weight=Math.exp(-x2*x2/2);weights.push(weight),i===0?sum+=weight:i<samples&&(sum+=2*weight)}for(let i=0;i<weights.length;i++)weights[i]=weights[i]/sum;blurUniforms.envMap.value=targetIn.texture,blurUniforms.samples.value=samples,blurUniforms.weights.value=weights,blurUniforms.latitudinal.value=direction==="latitudinal",poleAxis&&(blurUniforms.poleAxis.value=poleAxis);const{_lodMax}=this;blurUniforms.dTheta.value=radiansPerPixel,blurUniforms.mipInt.value=_lodMax-lodIn;const outputSize=this._sizeLods[lodOut],x=3*outputSize*(lodOut>_lodMax-LOD_MIN?lodOut-_lodMax+LOD_MIN:0),y=4*(this._cubeSize-outputSize);_setViewport(targetOut,x,y,3*outputSize,2*outputSize),renderer.setRenderTarget(targetOut),renderer.render(blurMesh,_flatCamera)}}function _createPlanes(lodMax){const lodPlanes=[],sizeLods=[],sigmas=[];let lod=lodMax;const totalLods=lodMax-LOD_MIN+1+EXTRA_LOD_SIGMA.length;for(let i=0;i<totalLods;i++){const sizeLod=Math.pow(2,lod);sizeLods.push(sizeLod);let sigma=1/sizeLod;i>lodMax-LOD_MIN?sigma=EXTRA_LOD_SIGMA[i-lodMax+LOD_MIN-1]:i===0&&(sigma=0),sigmas.push(sigma);const texelSize=1/(sizeLod-2),min=-texelSize,max2=1+texelSize,uv1=[min,min,max2,min,max2,max2,min,min,max2,max2,min,max2],cubeFaces=6,vertices=6,positionSize=3,uvSize=2,faceIndexSize=1,position=new Float32Array(positionSize*vertices*cubeFaces),uv=new Float32Array(uvSize*vertices*cubeFaces),faceIndex=new Float32Array(faceIndexSize*vertices*cubeFaces);for(let face=0;face<cubeFaces;face++){const x=face%3*2/3-1,y=face>2?0:-1,coordinates=[x,y,0,x+2/3,y,0,x+2/3,y+1,0,x,y,0,x+2/3,y+1,0,x,y+1,0];position.set(coordinates,positionSize*vertices*face),uv.set(uv1,uvSize*vertices*face);const fill=[face,face,face,face,face,face];faceIndex.set(fill,faceIndexSize*vertices*face)}const planes=new BufferGeometry;planes.setAttribute("position",new BufferAttribute(position,positionSize)),planes.setAttribute("uv",new BufferAttribute(uv,uvSize)),planes.setAttribute("faceIndex",new BufferAttribute(faceIndex,faceIndexSize)),lodPlanes.push(planes),lod>LOD_MIN&&lod--}return{lodPlanes,sizeLods,sigmas}}__name(_createPlanes,"_createPlanes");function _createRenderTarget(width,height,params){const cubeUVRenderTarget=new WebGLRenderTarget(width,height,params);return cubeUVRenderTarget.texture.mapping=CubeUVReflectionMapping,cubeUVRenderTarget.texture.name="PMREM.cubeUv",cubeUVRenderTarget.scissorTest=!0,cubeUVRenderTarget}__name(_createRenderTarget,"_createRenderTarget");function _setViewport(target,x,y,width,height){target.viewport.set(x,y,width,height),target.scissor.set(x,y,width,height)}__name(_setViewport,"_setViewport");function _getBlurShader(lodMax,width,height){const weights=new Float32Array(MAX_SAMPLES),poleAxis=new Vector3(0,1,0);return new ShaderMaterial({name:"SphericalGaussianBlur",defines:{n:MAX_SAMPLES,CUBEUV_TEXEL_WIDTH:1/width,CUBEUV_TEXEL_HEIGHT:1/height,CUBEUV_MAX_MIP:`${lodMax}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:weights},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:poleAxis}},vertexShader:_getCommonVertexShader(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:NoBlending,depthTest:!1,depthWrite:!1})}__name(_getBlurShader,"_getBlurShader");function _getEquirectMaterial(){return new ShaderMaterial({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:_getCommonVertexShader(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:NoBlending,depthTest:!1,depthWrite:!1})}__name(_getEquirectMaterial,"_getEquirectMaterial");function _getCubemapMaterial(){return new ShaderMaterial({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:_getCommonVertexShader(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:NoBlending,depthTest:!1,depthWrite:!1})}__name(_getCubemapMaterial,"_getCubemapMaterial");function _getCommonVertexShader(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}__name(_getCommonVertexShader,"_getCommonVertexShader");function WebGLCubeUVMaps(renderer){let cubeUVmaps=new WeakMap,pmremGenerator=null;function get(texture){if(texture&&texture.isTexture){const mapping=texture.mapping,isEquirectMap=mapping===EquirectangularReflectionMapping||mapping===EquirectangularRefractionMapping,isCubeMap=mapping===CubeReflectionMapping||mapping===CubeRefractionMapping;if(isEquirectMap||isCubeMap){let renderTarget=cubeUVmaps.get(texture);const currentPMREMVersion=renderTarget!==void 0?renderTarget.texture.pmremVersion:0;if(texture.isRenderTargetTexture&&texture.pmremVersion!==currentPMREMVersion)return pmremGenerator===null&&(pmremGenerator=new PMREMGenerator(renderer)),renderTarget=isEquirectMap?pmremGenerator.fromEquirectangular(texture,renderTarget):pmremGenerator.fromCubemap(texture,renderTarget),renderTarget.texture.pmremVersion=texture.pmremVersion,cubeUVmaps.set(texture,renderTarget),renderTarget.texture;if(renderTarget!==void 0)return renderTarget.texture;{const image=texture.image;return isEquirectMap&&image&&image.height>0||isCubeMap&&image&&isCubeTextureComplete(image)?(pmremGenerator===null&&(pmremGenerator=new PMREMGenerator(renderer)),renderTarget=isEquirectMap?pmremGenerator.fromEquirectangular(texture):pmremGenerator.fromCubemap(texture),renderTarget.texture.pmremVersion=texture.pmremVersion,cubeUVmaps.set(texture,renderTarget),texture.addEventListener("dispose",onTextureDispose),renderTarget.texture):null}}}return texture}__name(get,"get");function isCubeTextureComplete(image){let count=0;const length=6;for(let i=0;i<length;i++)image[i]!==void 0&&count++;return count===length}__name(isCubeTextureComplete,"isCubeTextureComplete");function onTextureDispose(event){const texture=event.target;texture.removeEventListener("dispose",onTextureDispose);const cubemapUV=cubeUVmaps.get(texture);cubemapUV!==void 0&&(cubeUVmaps.delete(texture),cubemapUV.dispose())}__name(onTextureDispose,"onTextureDispose");function dispose(){cubeUVmaps=new WeakMap,pmremGenerator!==null&&(pmremGenerator.dispose(),pmremGenerator=null)}return __name(dispose,"dispose"),{get,dispose}}__name(WebGLCubeUVMaps,"WebGLCubeUVMaps");function WebGLExtensions(gl){const extensions={};function getExtension(name){if(extensions[name]!==void 0)return extensions[name];let extension;switch(name){case"WEBGL_depth_texture":extension=gl.getExtension("WEBGL_depth_texture")||gl.getExtension("MOZ_WEBGL_depth_texture")||gl.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":extension=gl.getExtension("EXT_texture_filter_anisotropic")||gl.getExtension("MOZ_EXT_texture_filter_anisotropic")||gl.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":extension=gl.getExtension("WEBGL_compressed_texture_s3tc")||gl.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||gl.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":extension=gl.getExtension("WEBGL_compressed_texture_pvrtc")||gl.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:extension=gl.getExtension(name)}return extensions[name]=extension,extension}return __name(getExtension,"getExtension"),{has:__name(function(name){return getExtension(name)!==null},"has"),init:__name(function(){getExtension("EXT_color_buffer_float"),getExtension("WEBGL_clip_cull_distance"),getExtension("OES_texture_float_linear"),getExtension("EXT_color_buffer_half_float"),getExtension("WEBGL_multisampled_render_to_texture"),getExtension("WEBGL_render_shared_exponent")},"init"),get:__name(function(name){const extension=getExtension(name);return extension===null&&warnOnce("THREE.WebGLRenderer: "+name+" extension not supported."),extension},"get")}}__name(WebGLExtensions,"WebGLExtensions");function WebGLGeometries(gl,attributes,info,bindingStates){const geometries={},wireframeAttributes=new WeakMap;function onGeometryDispose(event){const geometry=event.target;geometry.index!==null&&attributes.remove(geometry.index);for(const name in geometry.attributes)attributes.remove(geometry.attributes[name]);for(const name in geometry.morphAttributes){const array=geometry.morphAttributes[name];for(let i=0,l=array.length;i<l;i++)attributes.remove(array[i])}geometry.removeEventListener("dispose",onGeometryDispose),delete geometries[geometry.id];const attribute=wireframeAttributes.get(geometry);attribute&&(attributes.remove(attribute),wireframeAttributes.delete(geometry)),bindingStates.releaseStatesOfGeometry(geometry),geometry.isInstancedBufferGeometry===!0&&delete geometry._maxInstanceCount,info.memory.geometries--}__name(onGeometryDispose,"onGeometryDispose");function get(object,geometry){return geometries[geometry.id]===!0||(geometry.addEventListener("dispose",onGeometryDispose),geometries[geometry.id]=!0,info.memory.geometries++),geometry}__name(get,"get");function update(geometry){const geometryAttributes=geometry.attributes;for(const name in geometryAttributes)attributes.update(geometryAttributes[name],gl.ARRAY_BUFFER);const morphAttributes=geometry.morphAttributes;for(const name in morphAttributes){const array=morphAttributes[name];for(let i=0,l=array.length;i<l;i++)attributes.update(array[i],gl.ARRAY_BUFFER)}}__name(update,"update");function updateWireframeAttribute(geometry){const indices=[],geometryIndex=geometry.index,geometryPosition=geometry.attributes.position;let version=0;if(geometryIndex!==null){const array=geometryIndex.array;version=geometryIndex.version;for(let i=0,l=array.length;i<l;i+=3){const a=array[i+0],b=array[i+1],c=array[i+2];indices.push(a,b,b,c,c,a)}}else if(geometryPosition!==void 0){const array=geometryPosition.array;version=geometryPosition.version;for(let i=0,l=array.length/3-1;i<l;i+=3){const a=i+0,b=i+1,c=i+2;indices.push(a,b,b,c,c,a)}}else return;const attribute=new(arrayNeedsUint32(indices)?Uint32BufferAttribute:Uint16BufferAttribute)(indices,1);attribute.version=version;const previousAttribute=wireframeAttributes.get(geometry);previousAttribute&&attributes.remove(previousAttribute),wireframeAttributes.set(geometry,attribute)}__name(updateWireframeAttribute,"updateWireframeAttribute");function getWireframeAttribute(geometry){const currentAttribute=wireframeAttributes.get(geometry);if(currentAttribute){const geometryIndex=geometry.index;geometryIndex!==null&&currentAttribute.version<geometryIndex.version&&updateWireframeAttribute(geometry)}else updateWireframeAttribute(geometry);return wireframeAttributes.get(geometry)}return __name(getWireframeAttribute,"getWireframeAttribute"),{get,update,getWireframeAttribute}}__name(WebGLGeometries,"WebGLGeometries");function WebGLIndexedBufferRenderer(gl,extensions,info){let mode;function setMode(value){mode=value}__name(setMode,"setMode");let type,bytesPerElement;function setIndex(value){type=value.type,bytesPerElement=value.bytesPerElement}__name(setIndex,"setIndex");function render(start,count){gl.drawElements(mode,count,type,start*bytesPerElement),info.update(count,mode,1)}__name(render,"render");function renderInstances(start,count,primcount){primcount!==0&&(gl.drawElementsInstanced(mode,count,type,start*bytesPerElement,primcount),info.update(count,mode,primcount))}__name(renderInstances,"renderInstances");function renderMultiDraw(starts,counts,drawCount){if(drawCount===0)return;extensions.get("WEBGL_multi_draw").multiDrawElementsWEBGL(mode,counts,0,type,starts,0,drawCount);let elementCount=0;for(let i=0;i<drawCount;i++)elementCount+=counts[i];info.update(elementCount,mode,1)}__name(renderMultiDraw,"renderMultiDraw");function renderMultiDrawInstances(starts,counts,drawCount,primcount){if(drawCount===0)return;const extension=extensions.get("WEBGL_multi_draw");if(extension===null)for(let i=0;i<starts.length;i++)renderInstances(starts[i]/bytesPerElement,counts[i],primcount[i]);else{extension.multiDrawElementsInstancedWEBGL(mode,counts,0,type,starts,0,primcount,0,drawCount);let elementCount=0;for(let i=0;i<drawCount;i++)elementCount+=counts[i]*primcount[i];info.update(elementCount,mode,1)}}__name(renderMultiDrawInstances,"renderMultiDrawInstances"),this.setMode=setMode,this.setIndex=setIndex,this.render=render,this.renderInstances=renderInstances,this.renderMultiDraw=renderMultiDraw,this.renderMultiDrawInstances=renderMultiDrawInstances}__name(WebGLIndexedBufferRenderer,"WebGLIndexedBufferRenderer");function WebGLInfo(gl){const memory={geometries:0,textures:0},render={frame:0,calls:0,triangles:0,points:0,lines:0};function update(count,mode,instanceCount){switch(render.calls++,mode){case gl.TRIANGLES:render.triangles+=instanceCount*(count/3);break;case gl.LINES:render.lines+=instanceCount*(count/2);break;case gl.LINE_STRIP:render.lines+=instanceCount*(count-1);break;case gl.LINE_LOOP:render.lines+=instanceCount*count;break;case gl.POINTS:render.points+=instanceCount*count;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",mode);break}}__name(update,"update");function reset(){render.calls=0,render.triangles=0,render.points=0,render.lines=0}return __name(reset,"reset"),{memory,render,programs:null,autoReset:!0,reset,update}}__name(WebGLInfo,"WebGLInfo");function WebGLMorphtargets(gl,capabilities,textures){const morphTextures=new WeakMap,morph=new Vector4;function update(object,geometry,program){const objectInfluences=object.morphTargetInfluences,morphAttribute=geometry.morphAttributes.position||geometry.morphAttributes.normal||geometry.morphAttributes.color,morphTargetsCount=morphAttribute!==void 0?morphAttribute.length:0;let entry=morphTextures.get(geometry);if(entry===void 0||entry.count!==morphTargetsCount){let disposeTexture=function(){texture.dispose(),morphTextures.delete(geometry),geometry.removeEventListener("dispose",disposeTexture)};__name(disposeTexture,"disposeTexture"),entry!==void 0&&entry.texture.dispose();const hasMorphPosition=geometry.morphAttributes.position!==void 0,hasMorphNormals=geometry.morphAttributes.normal!==void 0,hasMorphColors=geometry.morphAttributes.color!==void 0,morphTargets=geometry.morphAttributes.position||[],morphNormals=geometry.morphAttributes.normal||[],morphColors=geometry.morphAttributes.color||[];let vertexDataCount=0;hasMorphPosition===!0&&(vertexDataCount=1),hasMorphNormals===!0&&(vertexDataCount=2),hasMorphColors===!0&&(vertexDataCount=3);let width=geometry.attributes.position.count*vertexDataCount,height=1;width>capabilities.maxTextureSize&&(height=Math.ceil(width/capabilities.maxTextureSize),width=capabilities.maxTextureSize);const buffer=new Float32Array(width*height*4*morphTargetsCount),texture=new DataArrayTexture(buffer,width,height,morphTargetsCount);texture.type=FloatType,texture.needsUpdate=!0;const vertexDataStride=vertexDataCount*4;for(let i=0;i<morphTargetsCount;i++){const morphTarget=morphTargets[i],morphNormal=morphNormals[i],morphColor=morphColors[i],offset=width*height*4*i;for(let j=0;j<morphTarget.count;j++){const stride=j*vertexDataStride;hasMorphPosition===!0&&(morph.fromBufferAttribute(morphTarget,j),buffer[offset+stride+0]=morph.x,buffer[offset+stride+1]=morph.y,buffer[offset+stride+2]=morph.z,buffer[offset+stride+3]=0),hasMorphNormals===!0&&(morph.fromBufferAttribute(morphNormal,j),buffer[offset+stride+4]=morph.x,buffer[offset+stride+5]=morph.y,buffer[offset+stride+6]=morph.z,buffer[offset+stride+7]=0),hasMorphColors===!0&&(morph.fromBufferAttribute(morphColor,j),buffer[offset+stride+8]=morph.x,buffer[offset+stride+9]=morph.y,buffer[offset+stride+10]=morph.z,buffer[offset+stride+11]=morphColor.itemSize===4?morph.w:1)}}entry={count:morphTargetsCount,texture,size:new Vector2(width,height)},morphTextures.set(geometry,entry),geometry.addEventListener("dispose",disposeTexture)}if(object.isInstancedMesh===!0&&object.morphTexture!==null)program.getUniforms().setValue(gl,"morphTexture",object.morphTexture,textures);else{let morphInfluencesSum=0;for(let i=0;i<objectInfluences.length;i++)morphInfluencesSum+=objectInfluences[i];const morphBaseInfluence=geometry.morphTargetsRelative?1:1-morphInfluencesSum;program.getUniforms().setValue(gl,"morphTargetBaseInfluence",morphBaseInfluence),program.getUniforms().setValue(gl,"morphTargetInfluences",objectInfluences)}program.getUniforms().setValue(gl,"morphTargetsTexture",entry.texture,textures),program.getUniforms().setValue(gl,"morphTargetsTextureSize",entry.size)}return __name(update,"update"),{update}}__name(WebGLMorphtargets,"WebGLMorphtargets");function WebGLObjects(gl,geometries,attributes,info){let updateMap=new WeakMap;function update(object){const frame=info.render.frame,geometry=object.geometry,buffergeometry=geometries.get(object,geometry);if(updateMap.get(buffergeometry)!==frame&&(geometries.update(buffergeometry),updateMap.set(buffergeometry,frame)),object.isInstancedMesh&&(object.hasEventListener("dispose",onInstancedMeshDispose)===!1&&object.addEventListener("dispose",onInstancedMeshDispose),updateMap.get(object)!==frame&&(attributes.update(object.instanceMatrix,gl.ARRAY_BUFFER),object.instanceColor!==null&&attributes.update(object.instanceColor,gl.ARRAY_BUFFER),updateMap.set(object,frame))),object.isSkinnedMesh){const skeleton=object.skeleton;updateMap.get(skeleton)!==frame&&(skeleton.update(),updateMap.set(skeleton,frame))}return buffergeometry}__name(update,"update");function dispose(){updateMap=new WeakMap}__name(dispose,"dispose");function onInstancedMeshDispose(event){const instancedMesh=event.target;instancedMesh.removeEventListener("dispose",onInstancedMeshDispose),attributes.remove(instancedMesh.instanceMatrix),instancedMesh.instanceColor!==null&&attributes.remove(instancedMesh.instanceColor)}return __name(onInstancedMeshDispose,"onInstancedMeshDispose"),{update,dispose}}__name(WebGLObjects,"WebGLObjects");class DepthTexture extends Texture{static{__name(this,"DepthTexture")}constructor(width,height,type,mapping,wrapS,wrapT,magFilter,minFilter,anisotropy,format=DepthFormat){if(format!==DepthFormat&&format!==DepthStencilFormat)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");type===void 0&&format===DepthFormat&&(type=UnsignedIntType),type===void 0&&format===DepthStencilFormat&&(type=UnsignedInt248Type),super(null,mapping,wrapS,wrapT,magFilter,minFilter,format,type,anisotropy),this.isDepthTexture=!0,this.image={width,height},this.magFilter=magFilter!==void 0?magFilter:NearestFilter,this.minFilter=minFilter!==void 0?minFilter:NearestFilter,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(source){return super.copy(source),this.compareFunction=source.compareFunction,this}toJSON(meta){const data=super.toJSON(meta);return this.compareFunction!==null&&(data.compareFunction=this.compareFunction),data}}const emptyTexture=new Texture,emptyShadowTexture=new DepthTexture(1,1),emptyArrayTexture=new DataArrayTexture,empty3dTexture=new Data3DTexture,emptyCubeTexture=new CubeTexture,arrayCacheF32=[],arrayCacheI32=[],mat4array=new Float32Array(16),mat3array=new Float32Array(9),mat2array=new Float32Array(4);function flatten(array,nBlocks,blockSize){const firstElem=array[0];if(firstElem<=0||firstElem>0)return array;const n=nBlocks*blockSize;let r=arrayCacheF32[n];if(r===void 0&&(r=new Float32Array(n),arrayCacheF32[n]=r),nBlocks!==0){firstElem.toArray(r,0);for(let i=1,offset=0;i!==nBlocks;++i)offset+=blockSize,array[i].toArray(r,offset)}return r}__name(flatten,"flatten");function arraysEqual(a,b){if(a.length!==b.length)return!1;for(let i=0,l=a.length;i<l;i++)if(a[i]!==b[i])return!1;return!0}__name(arraysEqual,"arraysEqual");function copyArray(a,b){for(let i=0,l=b.length;i<l;i++)a[i]=b[i]}__name(copyArray,"copyArray");function allocTexUnits(textures,n){let r=arrayCacheI32[n];r===void 0&&(r=new Int32Array(n),arrayCacheI32[n]=r);for(let i=0;i!==n;++i)r[i]=textures.allocateTextureUnit();return r}__name(allocTexUnits,"allocTexUnits");function setValueV1f(gl,v){const cache=this.cache;cache[0]!==v&&(gl.uniform1f(this.addr,v),cache[0]=v)}__name(setValueV1f,"setValueV1f");function setValueV2f(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y)&&(gl.uniform2f(this.addr,v.x,v.y),cache[0]=v.x,cache[1]=v.y);else{if(arraysEqual(cache,v))return;gl.uniform2fv(this.addr,v),copyArray(cache,v)}}__name(setValueV2f,"setValueV2f");function setValueV3f(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y||cache[2]!==v.z)&&(gl.uniform3f(this.addr,v.x,v.y,v.z),cache[0]=v.x,cache[1]=v.y,cache[2]=v.z);else if(v.r!==void 0)(cache[0]!==v.r||cache[1]!==v.g||cache[2]!==v.b)&&(gl.uniform3f(this.addr,v.r,v.g,v.b),cache[0]=v.r,cache[1]=v.g,cache[2]=v.b);else{if(arraysEqual(cache,v))return;gl.uniform3fv(this.addr,v),copyArray(cache,v)}}__name(setValueV3f,"setValueV3f");function setValueV4f(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y||cache[2]!==v.z||cache[3]!==v.w)&&(gl.uniform4f(this.addr,v.x,v.y,v.z,v.w),cache[0]=v.x,cache[1]=v.y,cache[2]=v.z,cache[3]=v.w);else{if(arraysEqual(cache,v))return;gl.uniform4fv(this.addr,v),copyArray(cache,v)}}__name(setValueV4f,"setValueV4f");function setValueM2(gl,v){const cache=this.cache,elements=v.elements;if(elements===void 0){if(arraysEqual(cache,v))return;gl.uniformMatrix2fv(this.addr,!1,v),copyArray(cache,v)}else{if(arraysEqual(cache,elements))return;mat2array.set(elements),gl.uniformMatrix2fv(this.addr,!1,mat2array),copyArray(cache,elements)}}__name(setValueM2,"setValueM2");function setValueM3(gl,v){const cache=this.cache,elements=v.elements;if(elements===void 0){if(arraysEqual(cache,v))return;gl.uniformMatrix3fv(this.addr,!1,v),copyArray(cache,v)}else{if(arraysEqual(cache,elements))return;mat3array.set(elements),gl.uniformMatrix3fv(this.addr,!1,mat3array),copyArray(cache,elements)}}__name(setValueM3,"setValueM3");function setValueM4(gl,v){const cache=this.cache,elements=v.elements;if(elements===void 0){if(arraysEqual(cache,v))return;gl.uniformMatrix4fv(this.addr,!1,v),copyArray(cache,v)}else{if(arraysEqual(cache,elements))return;mat4array.set(elements),gl.uniformMatrix4fv(this.addr,!1,mat4array),copyArray(cache,elements)}}__name(setValueM4,"setValueM4");function setValueV1i(gl,v){const cache=this.cache;cache[0]!==v&&(gl.uniform1i(this.addr,v),cache[0]=v)}__name(setValueV1i,"setValueV1i");function setValueV2i(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y)&&(gl.uniform2i(this.addr,v.x,v.y),cache[0]=v.x,cache[1]=v.y);else{if(arraysEqual(cache,v))return;gl.uniform2iv(this.addr,v),copyArray(cache,v)}}__name(setValueV2i,"setValueV2i");function setValueV3i(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y||cache[2]!==v.z)&&(gl.uniform3i(this.addr,v.x,v.y,v.z),cache[0]=v.x,cache[1]=v.y,cache[2]=v.z);else{if(arraysEqual(cache,v))return;gl.uniform3iv(this.addr,v),copyArray(cache,v)}}__name(setValueV3i,"setValueV3i");function setValueV4i(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y||cache[2]!==v.z||cache[3]!==v.w)&&(gl.uniform4i(this.addr,v.x,v.y,v.z,v.w),cache[0]=v.x,cache[1]=v.y,cache[2]=v.z,cache[3]=v.w);else{if(arraysEqual(cache,v))return;gl.uniform4iv(this.addr,v),copyArray(cache,v)}}__name(setValueV4i,"setValueV4i");function setValueV1ui(gl,v){const cache=this.cache;cache[0]!==v&&(gl.uniform1ui(this.addr,v),cache[0]=v)}__name(setValueV1ui,"setValueV1ui");function setValueV2ui(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y)&&(gl.uniform2ui(this.addr,v.x,v.y),cache[0]=v.x,cache[1]=v.y);else{if(arraysEqual(cache,v))return;gl.uniform2uiv(this.addr,v),copyArray(cache,v)}}__name(setValueV2ui,"setValueV2ui");function setValueV3ui(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y||cache[2]!==v.z)&&(gl.uniform3ui(this.addr,v.x,v.y,v.z),cache[0]=v.x,cache[1]=v.y,cache[2]=v.z);else{if(arraysEqual(cache,v))return;gl.uniform3uiv(this.addr,v),copyArray(cache,v)}}__name(setValueV3ui,"setValueV3ui");function setValueV4ui(gl,v){const cache=this.cache;if(v.x!==void 0)(cache[0]!==v.x||cache[1]!==v.y||cache[2]!==v.z||cache[3]!==v.w)&&(gl.uniform4ui(this.addr,v.x,v.y,v.z,v.w),cache[0]=v.x,cache[1]=v.y,cache[2]=v.z,cache[3]=v.w);else{if(arraysEqual(cache,v))return;gl.uniform4uiv(this.addr,v),copyArray(cache,v)}}__name(setValueV4ui,"setValueV4ui");function setValueT1(gl,v,textures){const cache=this.cache,unit=textures.allocateTextureUnit();cache[0]!==unit&&(gl.uniform1i(this.addr,unit),cache[0]=unit);let emptyTexture2D;this.type===gl.SAMPLER_2D_SHADOW?(emptyShadowTexture.compareFunction=LessEqualCompare,emptyTexture2D=emptyShadowTexture):emptyTexture2D=emptyTexture,textures.setTexture2D(v||emptyTexture2D,unit)}__name(setValueT1,"setValueT1");function setValueT3D1(gl,v,textures){const cache=this.cache,unit=textures.allocateTextureUnit();cache[0]!==unit&&(gl.uniform1i(this.addr,unit),cache[0]=unit),textures.setTexture3D(v||empty3dTexture,unit)}__name(setValueT3D1,"setValueT3D1");function setValueT6(gl,v,textures){const cache=this.cache,unit=textures.allocateTextureUnit();cache[0]!==unit&&(gl.uniform1i(this.addr,unit),cache[0]=unit),textures.setTextureCube(v||emptyCubeTexture,unit)}__name(setValueT6,"setValueT6");function setValueT2DArray1(gl,v,textures){const cache=this.cache,unit=textures.allocateTextureUnit();cache[0]!==unit&&(gl.uniform1i(this.addr,unit),cache[0]=unit),textures.setTexture2DArray(v||emptyArrayTexture,unit)}__name(setValueT2DArray1,"setValueT2DArray1");function getSingularSetter(type){switch(type){case 5126:return setValueV1f;case 35664:return setValueV2f;case 35665:return setValueV3f;case 35666:return setValueV4f;case 35674:return setValueM2;case 35675:return setValueM3;case 35676:return setValueM4;case 5124:case 35670:return setValueV1i;case 35667:case 35671:return setValueV2i;case 35668:case 35672:return setValueV3i;case 35669:case 35673:return setValueV4i;case 5125:return setValueV1ui;case 36294:return setValueV2ui;case 36295:return setValueV3ui;case 36296:return setValueV4ui;case 35678:case 36198:case 36298:case 36306:case 35682:return setValueT1;case 35679:case 36299:case 36307:return setValueT3D1;case 35680:case 36300:case 36308:case 36293:return setValueT6;case 36289:case 36303:case 36311:case 36292:return setValueT2DArray1}}__name(getSingularSetter,"getSingularSetter");function setValueV1fArray(gl,v){gl.uniform1fv(this.addr,v)}__name(setValueV1fArray,"setValueV1fArray");function setValueV2fArray(gl,v){const data=flatten(v,this.size,2);gl.uniform2fv(this.addr,data)}__name(setValueV2fArray,"setValueV2fArray");function setValueV3fArray(gl,v){const data=flatten(v,this.size,3);gl.uniform3fv(this.addr,data)}__name(setValueV3fArray,"setValueV3fArray");function setValueV4fArray(gl,v){const data=flatten(v,this.size,4);gl.uniform4fv(this.addr,data)}__name(setValueV4fArray,"setValueV4fArray");function setValueM2Array(gl,v){const data=flatten(v,this.size,4);gl.uniformMatrix2fv(this.addr,!1,data)}__name(setValueM2Array,"setValueM2Array");function setValueM3Array(gl,v){const data=flatten(v,this.size,9);gl.uniformMatrix3fv(this.addr,!1,data)}__name(setValueM3Array,"setValueM3Array");function setValueM4Array(gl,v){const data=flatten(v,this.size,16);gl.uniformMatrix4fv(this.addr,!1,data)}__name(setValueM4Array,"setValueM4Array");function setValueV1iArray(gl,v){gl.uniform1iv(this.addr,v)}__name(setValueV1iArray,"setValueV1iArray");function setValueV2iArray(gl,v){gl.uniform2iv(this.addr,v)}__name(setValueV2iArray,"setValueV2iArray");function setValueV3iArray(gl,v){gl.uniform3iv(this.addr,v)}__name(setValueV3iArray,"setValueV3iArray");function setValueV4iArray(gl,v){gl.uniform4iv(this.addr,v)}__name(setValueV4iArray,"setValueV4iArray");function setValueV1uiArray(gl,v){gl.uniform1uiv(this.addr,v)}__name(setValueV1uiArray,"setValueV1uiArray");function setValueV2uiArray(gl,v){gl.uniform2uiv(this.addr,v)}__name(setValueV2uiArray,"setValueV2uiArray");function setValueV3uiArray(gl,v){gl.uniform3uiv(this.addr,v)}__name(setValueV3uiArray,"setValueV3uiArray");function setValueV4uiArray(gl,v){gl.uniform4uiv(this.addr,v)}__name(setValueV4uiArray,"setValueV4uiArray");function setValueT1Array(gl,v,textures){const cache=this.cache,n=v.length,units=allocTexUnits(textures,n);arraysEqual(cache,units)||(gl.uniform1iv(this.addr,units),copyArray(cache,units));for(let i=0;i!==n;++i)textures.setTexture2D(v[i]||emptyTexture,units[i])}__name(setValueT1Array,"setValueT1Array");function setValueT3DArray(gl,v,textures){const cache=this.cache,n=v.length,units=allocTexUnits(textures,n);arraysEqual(cache,units)||(gl.uniform1iv(this.addr,units),copyArray(cache,units));for(let i=0;i!==n;++i)textures.setTexture3D(v[i]||empty3dTexture,units[i])}__name(setValueT3DArray,"setValueT3DArray");function setValueT6Array(gl,v,textures){const cache=this.cache,n=v.length,units=allocTexUnits(textures,n);arraysEqual(cache,units)||(gl.uniform1iv(this.addr,units),copyArray(cache,units));for(let i=0;i!==n;++i)textures.setTextureCube(v[i]||emptyCubeTexture,units[i])}__name(setValueT6Array,"setValueT6Array");function setValueT2DArrayArray(gl,v,textures){const cache=this.cache,n=v.length,units=allocTexUnits(textures,n);arraysEqual(cache,units)||(gl.uniform1iv(this.addr,units),copyArray(cache,units));for(let i=0;i!==n;++i)textures.setTexture2DArray(v[i]||emptyArrayTexture,units[i])}__name(setValueT2DArrayArray,"setValueT2DArrayArray");function getPureArraySetter(type){switch(type){case 5126:return setValueV1fArray;case 35664:return setValueV2fArray;case 35665:return setValueV3fArray;case 35666:return setValueV4fArray;case 35674:return setValueM2Array;case 35675:return setValueM3Array;case 35676:return setValueM4Array;case 5124:case 35670:return setValueV1iArray;case 35667:case 35671:return setValueV2iArray;case 35668:case 35672:return setValueV3iArray;case 35669:case 35673:return setValueV4iArray;case 5125:return setValueV1uiArray;case 36294:return setValueV2uiArray;case 36295:return setValueV3uiArray;case 36296:return setValueV4uiArray;case 35678:case 36198:case 36298:case 36306:case 35682:return setValueT1Array;case 35679:case 36299:case 36307:return setValueT3DArray;case 35680:case 36300:case 36308:case 36293:return setValueT6Array;case 36289:case 36303:case 36311:case 36292:return setValueT2DArrayArray}}__name(getPureArraySetter,"getPureArraySetter");class SingleUniform{static{__name(this,"SingleUniform")}constructor(id2,activeInfo,addr){this.id=id2,this.addr=addr,this.cache=[],this.type=activeInfo.type,this.setValue=getSingularSetter(activeInfo.type)}}class PureArrayUniform{static{__name(this,"PureArrayUniform")}constructor(id2,activeInfo,addr){this.id=id2,this.addr=addr,this.cache=[],this.type=activeInfo.type,this.size=activeInfo.size,this.setValue=getPureArraySetter(activeInfo.type)}}class StructuredUniform{static{__name(this,"StructuredUniform")}constructor(id2){this.id=id2,this.seq=[],this.map={}}setValue(gl,value,textures){const seq=this.seq;for(let i=0,n=seq.length;i!==n;++i){const u=seq[i];u.setValue(gl,value[u.id],textures)}}}const RePathPart=/(\w+)(\])?(\[|\.)?/g;function addUniform(container,uniformObject){container.seq.push(uniformObject),container.map[uniformObject.id]=uniformObject}__name(addUniform,"addUniform");function parseUniform(activeInfo,addr,container){const path=activeInfo.name,pathLength=path.length;for(RePathPart.lastIndex=0;;){const match=RePathPart.exec(path),matchEnd=RePathPart.lastIndex;let id2=match[1];const idIsIndex=match[2]==="]",subscript=match[3];if(idIsIndex&&(id2=id2|0),subscript===void 0||subscript==="["&&matchEnd+2===pathLength){addUniform(container,subscript===void 0?new SingleUniform(id2,activeInfo,addr):new PureArrayUniform(id2,activeInfo,addr));break}else{let next=container.map[id2];next===void 0&&(next=new StructuredUniform(id2),addUniform(container,next)),container=next}}}__name(parseUniform,"parseUniform");class WebGLUniforms{static{__name(this,"WebGLUniforms")}constructor(gl,program){this.seq=[],this.map={};const n=gl.getProgramParameter(program,gl.ACTIVE_UNIFORMS);for(let i=0;i<n;++i){const info=gl.getActiveUniform(program,i),addr=gl.getUniformLocation(program,info.name);parseUniform(info,addr,this)}}setValue(gl,name,value,textures){const u=this.map[name];u!==void 0&&u.setValue(gl,value,textures)}setOptional(gl,object,name){const v=object[name];v!==void 0&&this.setValue(gl,name,v)}static upload(gl,seq,values,textures){for(let i=0,n=seq.length;i!==n;++i){const u=seq[i],v=values[u.id];v.needsUpdate!==!1&&u.setValue(gl,v.value,textures)}}static seqWithValue(seq,values){const r=[];for(let i=0,n=seq.length;i!==n;++i){const u=seq[i];u.id in values&&r.push(u)}return r}}function WebGLShader(gl,type,string){const shader=gl.createShader(type);return gl.shaderSource(shader,string),gl.compileShader(shader),shader}__name(WebGLShader,"WebGLShader");const COMPLETION_STATUS_KHR=37297;let programIdCount=0;function handleSource(string,errorLine){const lines=string.split(`
`),lines2=[],from=Math.max(errorLine-6,0),to=Math.min(errorLine+6,lines.length);for(let i=from;i<to;i++){const line=i+1;lines2.push(`${line===errorLine?">":" "} ${line}: ${lines[i]}`)}return lines2.join(`
`)}__name(handleSource,"handleSource");const _m0=new Matrix3;function getEncodingComponents(colorSpace){ColorManagement._getMatrix(_m0,ColorManagement.workingColorSpace,colorSpace);const encodingMatrix=`mat3( ${_m0.elements.map(v=>v.toFixed(4))} )`;switch(ColorManagement.getTransfer(colorSpace)){case LinearTransfer:return[encodingMatrix,"LinearTransferOETF"];case SRGBTransfer:return[encodingMatrix,"sRGBTransferOETF"];default:return console.warn("THREE.WebGLProgram: Unsupported color space: ",colorSpace),[encodingMatrix,"LinearTransferOETF"]}}__name(getEncodingComponents,"getEncodingComponents");function getShaderErrors(gl,shader,type){const status=gl.getShaderParameter(shader,gl.COMPILE_STATUS),errors=gl.getShaderInfoLog(shader).trim();if(status&&errors==="")return"";const errorMatches=/ERROR: 0:(\d+)/.exec(errors);if(errorMatches){const errorLine=parseInt(errorMatches[1]);return type.toUpperCase()+`

`+errors+`

`+handleSource(gl.getShaderSource(shader),errorLine)}else return errors}__name(getShaderErrors,"getShaderErrors");function getTexelEncodingFunction(functionName,colorSpace){const components=getEncodingComponents(colorSpace);return[`vec4 ${functionName}( vec4 value ) {`,`	return ${components[1]}( vec4( value.rgb * ${components[0]}, value.a ) );`,"}"].join(`
`)}__name(getTexelEncodingFunction,"getTexelEncodingFunction");function getToneMappingFunction(functionName,toneMapping){let toneMappingName;switch(toneMapping){case LinearToneMapping:toneMappingName="Linear";break;case ReinhardToneMapping:toneMappingName="Reinhard";break;case CineonToneMapping:toneMappingName="Cineon";break;case ACESFilmicToneMapping:toneMappingName="ACESFilmic";break;case AgXToneMapping:toneMappingName="AgX";break;case NeutralToneMapping:toneMappingName="Neutral";break;case CustomToneMapping:toneMappingName="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",toneMapping),toneMappingName="Linear"}return"vec3 "+functionName+"( vec3 color ) { return "+toneMappingName+"ToneMapping( color ); }"}__name(getToneMappingFunction,"getToneMappingFunction");const _v0$1=new Vector3;function getLuminanceFunction(){ColorManagement.getLuminanceCoefficients(_v0$1);const r=_v0$1.x.toFixed(4),g=_v0$1.y.toFixed(4),b=_v0$1.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${r}, ${g}, ${b} );`,"	return dot( weights, rgb );","}"].join(`
`)}__name(getLuminanceFunction,"getLuminanceFunction");function generateVertexExtensions(parameters){return[parameters.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",parameters.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(filterEmptyLine).join(`
`)}__name(generateVertexExtensions,"generateVertexExtensions");function generateDefines(defines){const chunks=[];for(const name in defines){const value=defines[name];value!==!1&&chunks.push("#define "+name+" "+value)}return chunks.join(`
`)}__name(generateDefines,"generateDefines");function fetchAttributeLocations(gl,program){const attributes={},n=gl.getProgramParameter(program,gl.ACTIVE_ATTRIBUTES);for(let i=0;i<n;i++){const info=gl.getActiveAttrib(program,i),name=info.name;let locationSize=1;info.type===gl.FLOAT_MAT2&&(locationSize=2),info.type===gl.FLOAT_MAT3&&(locationSize=3),info.type===gl.FLOAT_MAT4&&(locationSize=4),attributes[name]={type:info.type,location:gl.getAttribLocation(program,name),locationSize}}return attributes}__name(fetchAttributeLocations,"fetchAttributeLocations");function filterEmptyLine(string){return string!==""}__name(filterEmptyLine,"filterEmptyLine");function replaceLightNums(string,parameters){const numSpotLightCoords=parameters.numSpotLightShadows+parameters.numSpotLightMaps-parameters.numSpotLightShadowsWithMaps;return string.replace(/NUM_DIR_LIGHTS/g,parameters.numDirLights).replace(/NUM_SPOT_LIGHTS/g,parameters.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,parameters.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,numSpotLightCoords).replace(/NUM_RECT_AREA_LIGHTS/g,parameters.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,parameters.numPointLights).replace(/NUM_HEMI_LIGHTS/g,parameters.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,parameters.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,parameters.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,parameters.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,parameters.numPointLightShadows)}__name(replaceLightNums,"replaceLightNums");function replaceClippingPlaneNums(string,parameters){return string.replace(/NUM_CLIPPING_PLANES/g,parameters.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,parameters.numClippingPlanes-parameters.numClipIntersection)}__name(replaceClippingPlaneNums,"replaceClippingPlaneNums");const includePattern=/^[ \t]*#include +<([\w\d./]+)>/gm;function resolveIncludes(string){return string.replace(includePattern,includeReplacer)}__name(resolveIncludes,"resolveIncludes");const shaderChunkMap=new Map;function includeReplacer(match,include){let string=ShaderChunk[include];if(string===void 0){const newInclude=shaderChunkMap.get(include);if(newInclude!==void 0)string=ShaderChunk[newInclude],console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',include,newInclude);else throw new Error("Can not resolve #include <"+include+">")}return resolveIncludes(string)}__name(includeReplacer,"includeReplacer");const unrollLoopPattern=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function unrollLoops(string){return string.replace(unrollLoopPattern,loopReplacer)}__name(unrollLoops,"unrollLoops");function loopReplacer(match,start,end,snippet){let string="";for(let i=parseInt(start);i<parseInt(end);i++)string+=snippet.replace(/\[\s*i\s*\]/g,"[ "+i+" ]").replace(/UNROLLED_LOOP_INDEX/g,i);return string}__name(loopReplacer,"loopReplacer");function generatePrecision(parameters){let precisionstring=`precision ${parameters.precision} float;
	precision ${parameters.precision} int;
	precision ${parameters.precision} sampler2D;
	precision ${parameters.precision} samplerCube;
	precision ${parameters.precision} sampler3D;
	precision ${parameters.precision} sampler2DArray;
	precision ${parameters.precision} sampler2DShadow;
	precision ${parameters.precision} samplerCubeShadow;
	precision ${parameters.precision} sampler2DArrayShadow;
	precision ${parameters.precision} isampler2D;
	precision ${parameters.precision} isampler3D;
	precision ${parameters.precision} isamplerCube;
	precision ${parameters.precision} isampler2DArray;
	precision ${parameters.precision} usampler2D;
	precision ${parameters.precision} usampler3D;
	precision ${parameters.precision} usamplerCube;
	precision ${parameters.precision} usampler2DArray;
	`;return parameters.precision==="highp"?precisionstring+=`
#define HIGH_PRECISION`:parameters.precision==="mediump"?precisionstring+=`
#define MEDIUM_PRECISION`:parameters.precision==="lowp"&&(precisionstring+=`
#define LOW_PRECISION`),precisionstring}__name(generatePrecision,"generatePrecision");function generateShadowMapTypeDefine(parameters){let shadowMapTypeDefine="SHADOWMAP_TYPE_BASIC";return parameters.shadowMapType===PCFShadowMap?shadowMapTypeDefine="SHADOWMAP_TYPE_PCF":parameters.shadowMapType===PCFSoftShadowMap?shadowMapTypeDefine="SHADOWMAP_TYPE_PCF_SOFT":parameters.shadowMapType===VSMShadowMap&&(shadowMapTypeDefine="SHADOWMAP_TYPE_VSM"),shadowMapTypeDefine}__name(generateShadowMapTypeDefine,"generateShadowMapTypeDefine");function generateEnvMapTypeDefine(parameters){let envMapTypeDefine="ENVMAP_TYPE_CUBE";if(parameters.envMap)switch(parameters.envMapMode){case CubeReflectionMapping:case CubeRefractionMapping:envMapTypeDefine="ENVMAP_TYPE_CUBE";break;case CubeUVReflectionMapping:envMapTypeDefine="ENVMAP_TYPE_CUBE_UV";break}return envMapTypeDefine}__name(generateEnvMapTypeDefine,"generateEnvMapTypeDefine");function generateEnvMapModeDefine(parameters){let envMapModeDefine="ENVMAP_MODE_REFLECTION";if(parameters.envMap)switch(parameters.envMapMode){case CubeRefractionMapping:envMapModeDefine="ENVMAP_MODE_REFRACTION";break}return envMapModeDefine}__name(generateEnvMapModeDefine,"generateEnvMapModeDefine");function generateEnvMapBlendingDefine(parameters){let envMapBlendingDefine="ENVMAP_BLENDING_NONE";if(parameters.envMap)switch(parameters.combine){case MultiplyOperation:envMapBlendingDefine="ENVMAP_BLENDING_MULTIPLY";break;case MixOperation:envMapBlendingDefine="ENVMAP_BLENDING_MIX";break;case AddOperation:envMapBlendingDefine="ENVMAP_BLENDING_ADD";break}return envMapBlendingDefine}__name(generateEnvMapBlendingDefine,"generateEnvMapBlendingDefine");function generateCubeUVSize(parameters){const imageHeight=parameters.envMapCubeUVHeight;if(imageHeight===null)return null;const maxMip=Math.log2(imageHeight)-2,texelHeight=1/imageHeight;return{texelWidth:1/(3*Math.max(Math.pow(2,maxMip),7*16)),texelHeight,maxMip}}__name(generateCubeUVSize,"generateCubeUVSize");function WebGLProgram(renderer,cacheKey,parameters,bindingStates){const gl=renderer.getContext(),defines=parameters.defines;let vertexShader=parameters.vertexShader,fragmentShader=parameters.fragmentShader;const shadowMapTypeDefine=generateShadowMapTypeDefine(parameters),envMapTypeDefine=generateEnvMapTypeDefine(parameters),envMapModeDefine=generateEnvMapModeDefine(parameters),envMapBlendingDefine=generateEnvMapBlendingDefine(parameters),envMapCubeUVSize=generateCubeUVSize(parameters),customVertexExtensions=generateVertexExtensions(parameters),customDefines=generateDefines(defines),program=gl.createProgram();let prefixVertex,prefixFragment,versionString=parameters.glslVersion?"#version "+parameters.glslVersion+`
`:"";parameters.isRawShaderMaterial?(prefixVertex=["#define SHADER_TYPE "+parameters.shaderType,"#define SHADER_NAME "+parameters.shaderName,customDefines].filter(filterEmptyLine).join(`
`),prefixVertex.length>0&&(prefixVertex+=`
`),prefixFragment=["#define SHADER_TYPE "+parameters.shaderType,"#define SHADER_NAME "+parameters.shaderName,customDefines].filter(filterEmptyLine).join(`
`),prefixFragment.length>0&&(prefixFragment+=`
`)):(prefixVertex=[generatePrecision(parameters),"#define SHADER_TYPE "+parameters.shaderType,"#define SHADER_NAME "+parameters.shaderName,customDefines,parameters.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",parameters.batching?"#define USE_BATCHING":"",parameters.batchingColor?"#define USE_BATCHING_COLOR":"",parameters.instancing?"#define USE_INSTANCING":"",parameters.instancingColor?"#define USE_INSTANCING_COLOR":"",parameters.instancingMorph?"#define USE_INSTANCING_MORPH":"",parameters.useFog&&parameters.fog?"#define USE_FOG":"",parameters.useFog&&parameters.fogExp2?"#define FOG_EXP2":"",parameters.map?"#define USE_MAP":"",parameters.envMap?"#define USE_ENVMAP":"",parameters.envMap?"#define "+envMapModeDefine:"",parameters.lightMap?"#define USE_LIGHTMAP":"",parameters.aoMap?"#define USE_AOMAP":"",parameters.bumpMap?"#define USE_BUMPMAP":"",parameters.normalMap?"#define USE_NORMALMAP":"",parameters.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",parameters.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",parameters.displacementMap?"#define USE_DISPLACEMENTMAP":"",parameters.emissiveMap?"#define USE_EMISSIVEMAP":"",parameters.anisotropy?"#define USE_ANISOTROPY":"",parameters.anisotropyMap?"#define USE_ANISOTROPYMAP":"",parameters.clearcoatMap?"#define USE_CLEARCOATMAP":"",parameters.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",parameters.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",parameters.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",parameters.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",parameters.specularMap?"#define USE_SPECULARMAP":"",parameters.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",parameters.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",parameters.roughnessMap?"#define USE_ROUGHNESSMAP":"",parameters.metalnessMap?"#define USE_METALNESSMAP":"",parameters.alphaMap?"#define USE_ALPHAMAP":"",parameters.alphaHash?"#define USE_ALPHAHASH":"",parameters.transmission?"#define USE_TRANSMISSION":"",parameters.transmissionMap?"#define USE_TRANSMISSIONMAP":"",parameters.thicknessMap?"#define USE_THICKNESSMAP":"",parameters.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",parameters.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",parameters.mapUv?"#define MAP_UV "+parameters.mapUv:"",parameters.alphaMapUv?"#define ALPHAMAP_UV "+parameters.alphaMapUv:"",parameters.lightMapUv?"#define LIGHTMAP_UV "+parameters.lightMapUv:"",parameters.aoMapUv?"#define AOMAP_UV "+parameters.aoMapUv:"",parameters.emissiveMapUv?"#define EMISSIVEMAP_UV "+parameters.emissiveMapUv:"",parameters.bumpMapUv?"#define BUMPMAP_UV "+parameters.bumpMapUv:"",parameters.normalMapUv?"#define NORMALMAP_UV "+parameters.normalMapUv:"",parameters.displacementMapUv?"#define DISPLACEMENTMAP_UV "+parameters.displacementMapUv:"",parameters.metalnessMapUv?"#define METALNESSMAP_UV "+parameters.metalnessMapUv:"",parameters.roughnessMapUv?"#define ROUGHNESSMAP_UV "+parameters.roughnessMapUv:"",parameters.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+parameters.anisotropyMapUv:"",parameters.clearcoatMapUv?"#define CLEARCOATMAP_UV "+parameters.clearcoatMapUv:"",parameters.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+parameters.clearcoatNormalMapUv:"",parameters.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+parameters.clearcoatRoughnessMapUv:"",parameters.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+parameters.iridescenceMapUv:"",parameters.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+parameters.iridescenceThicknessMapUv:"",parameters.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+parameters.sheenColorMapUv:"",parameters.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+parameters.sheenRoughnessMapUv:"",parameters.specularMapUv?"#define SPECULARMAP_UV "+parameters.specularMapUv:"",parameters.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+parameters.specularColorMapUv:"",parameters.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+parameters.specularIntensityMapUv:"",parameters.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+parameters.transmissionMapUv:"",parameters.thicknessMapUv?"#define THICKNESSMAP_UV "+parameters.thicknessMapUv:"",parameters.vertexTangents&&parameters.flatShading===!1?"#define USE_TANGENT":"",parameters.vertexColors?"#define USE_COLOR":"",parameters.vertexAlphas?"#define USE_COLOR_ALPHA":"",parameters.vertexUv1s?"#define USE_UV1":"",parameters.vertexUv2s?"#define USE_UV2":"",parameters.vertexUv3s?"#define USE_UV3":"",parameters.pointsUvs?"#define USE_POINTS_UV":"",parameters.flatShading?"#define FLAT_SHADED":"",parameters.skinning?"#define USE_SKINNING":"",parameters.morphTargets?"#define USE_MORPHTARGETS":"",parameters.morphNormals&&parameters.flatShading===!1?"#define USE_MORPHNORMALS":"",parameters.morphColors?"#define USE_MORPHCOLORS":"",parameters.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+parameters.morphTextureStride:"",parameters.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+parameters.morphTargetsCount:"",parameters.doubleSided?"#define DOUBLE_SIDED":"",parameters.flipSided?"#define FLIP_SIDED":"",parameters.shadowMapEnabled?"#define USE_SHADOWMAP":"",parameters.shadowMapEnabled?"#define "+shadowMapTypeDefine:"",parameters.sizeAttenuation?"#define USE_SIZEATTENUATION":"",parameters.numLightProbes>0?"#define USE_LIGHT_PROBES":"",parameters.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",parameters.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(filterEmptyLine).join(`
`),prefixFragment=[generatePrecision(parameters),"#define SHADER_TYPE "+parameters.shaderType,"#define SHADER_NAME "+parameters.shaderName,customDefines,parameters.useFog&&parameters.fog?"#define USE_FOG":"",parameters.useFog&&parameters.fogExp2?"#define FOG_EXP2":"",parameters.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",parameters.map?"#define USE_MAP":"",parameters.matcap?"#define USE_MATCAP":"",parameters.envMap?"#define USE_ENVMAP":"",parameters.envMap?"#define "+envMapTypeDefine:"",parameters.envMap?"#define "+envMapModeDefine:"",parameters.envMap?"#define "+envMapBlendingDefine:"",envMapCubeUVSize?"#define CUBEUV_TEXEL_WIDTH "+envMapCubeUVSize.texelWidth:"",envMapCubeUVSize?"#define CUBEUV_TEXEL_HEIGHT "+envMapCubeUVSize.texelHeight:"",envMapCubeUVSize?"#define CUBEUV_MAX_MIP "+envMapCubeUVSize.maxMip+".0":"",parameters.lightMap?"#define USE_LIGHTMAP":"",parameters.aoMap?"#define USE_AOMAP":"",parameters.bumpMap?"#define USE_BUMPMAP":"",parameters.normalMap?"#define USE_NORMALMAP":"",parameters.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",parameters.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",parameters.emissiveMap?"#define USE_EMISSIVEMAP":"",parameters.anisotropy?"#define USE_ANISOTROPY":"",parameters.anisotropyMap?"#define USE_ANISOTROPYMAP":"",parameters.clearcoat?"#define USE_CLEARCOAT":"",parameters.clearcoatMap?"#define USE_CLEARCOATMAP":"",parameters.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",parameters.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",parameters.dispersion?"#define USE_DISPERSION":"",parameters.iridescence?"#define USE_IRIDESCENCE":"",parameters.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",parameters.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",parameters.specularMap?"#define USE_SPECULARMAP":"",parameters.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",parameters.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",parameters.roughnessMap?"#define USE_ROUGHNESSMAP":"",parameters.metalnessMap?"#define USE_METALNESSMAP":"",parameters.alphaMap?"#define USE_ALPHAMAP":"",parameters.alphaTest?"#define USE_ALPHATEST":"",parameters.alphaHash?"#define USE_ALPHAHASH":"",parameters.sheen?"#define USE_SHEEN":"",parameters.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",parameters.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",parameters.transmission?"#define USE_TRANSMISSION":"",parameters.transmissionMap?"#define USE_TRANSMISSIONMAP":"",parameters.thicknessMap?"#define USE_THICKNESSMAP":"",parameters.vertexTangents&&parameters.flatShading===!1?"#define USE_TANGENT":"",parameters.vertexColors||parameters.instancingColor||parameters.batchingColor?"#define USE_COLOR":"",parameters.vertexAlphas?"#define USE_COLOR_ALPHA":"",parameters.vertexUv1s?"#define USE_UV1":"",parameters.vertexUv2s?"#define USE_UV2":"",parameters.vertexUv3s?"#define USE_UV3":"",parameters.pointsUvs?"#define USE_POINTS_UV":"",parameters.gradientMap?"#define USE_GRADIENTMAP":"",parameters.flatShading?"#define FLAT_SHADED":"",parameters.doubleSided?"#define DOUBLE_SIDED":"",parameters.flipSided?"#define FLIP_SIDED":"",parameters.shadowMapEnabled?"#define USE_SHADOWMAP":"",parameters.shadowMapEnabled?"#define "+shadowMapTypeDefine:"",parameters.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",parameters.numLightProbes>0?"#define USE_LIGHT_PROBES":"",parameters.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",parameters.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",parameters.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",parameters.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",parameters.toneMapping!==NoToneMapping?"#define TONE_MAPPING":"",parameters.toneMapping!==NoToneMapping?ShaderChunk.tonemapping_pars_fragment:"",parameters.toneMapping!==NoToneMapping?getToneMappingFunction("toneMapping",parameters.toneMapping):"",parameters.dithering?"#define DITHERING":"",parameters.opaque?"#define OPAQUE":"",ShaderChunk.colorspace_pars_fragment,getTexelEncodingFunction("linearToOutputTexel",parameters.outputColorSpace),getLuminanceFunction(),parameters.useDepthPacking?"#define DEPTH_PACKING "+parameters.depthPacking:"",`
`].filter(filterEmptyLine).join(`
`)),vertexShader=resolveIncludes(vertexShader),vertexShader=replaceLightNums(vertexShader,parameters),vertexShader=replaceClippingPlaneNums(vertexShader,parameters),fragmentShader=resolveIncludes(fragmentShader),fragmentShader=replaceLightNums(fragmentShader,parameters),fragmentShader=replaceClippingPlaneNums(fragmentShader,parameters),vertexShader=unrollLoops(vertexShader),fragmentShader=unrollLoops(fragmentShader),parameters.isRawShaderMaterial!==!0&&(versionString=`#version 300 es
`,prefixVertex=[customVertexExtensions,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+prefixVertex,prefixFragment=["#define varying in",parameters.glslVersion===GLSL3?"":"layout(location = 0) out highp vec4 pc_fragColor;",parameters.glslVersion===GLSL3?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+prefixFragment);const vertexGlsl=versionString+prefixVertex+vertexShader,fragmentGlsl=versionString+prefixFragment+fragmentShader,glVertexShader=WebGLShader(gl,gl.VERTEX_SHADER,vertexGlsl),glFragmentShader=WebGLShader(gl,gl.FRAGMENT_SHADER,fragmentGlsl);gl.attachShader(program,glVertexShader),gl.attachShader(program,glFragmentShader),parameters.index0AttributeName!==void 0?gl.bindAttribLocation(program,0,parameters.index0AttributeName):parameters.morphTargets===!0&&gl.bindAttribLocation(program,0,"position"),gl.linkProgram(program);function onFirstUse(self2){if(renderer.debug.checkShaderErrors){const programLog=gl.getProgramInfoLog(program).trim(),vertexLog=gl.getShaderInfoLog(glVertexShader).trim(),fragmentLog=gl.getShaderInfoLog(glFragmentShader).trim();let runnable=!0,haveDiagnostics=!0;if(gl.getProgramParameter(program,gl.LINK_STATUS)===!1)if(runnable=!1,typeof renderer.debug.onShaderError=="function")renderer.debug.onShaderError(gl,program,glVertexShader,glFragmentShader);else{const vertexErrors=getShaderErrors(gl,glVertexShader,"vertex"),fragmentErrors=getShaderErrors(gl,glFragmentShader,"fragment");console.error("THREE.WebGLProgram: Shader Error "+gl.getError()+" - VALIDATE_STATUS "+gl.getProgramParameter(program,gl.VALIDATE_STATUS)+`

Material Name: `+self2.name+`
Material Type: `+self2.type+`

Program Info Log: `+programLog+`
`+vertexErrors+`
`+fragmentErrors)}else programLog!==""?console.warn("THREE.WebGLProgram: Program Info Log:",programLog):(vertexLog===""||fragmentLog==="")&&(haveDiagnostics=!1);haveDiagnostics&&(self2.diagnostics={runnable,programLog,vertexShader:{log:vertexLog,prefix:prefixVertex},fragmentShader:{log:fragmentLog,prefix:prefixFragment}})}gl.deleteShader(glVertexShader),gl.deleteShader(glFragmentShader),cachedUniforms=new WebGLUniforms(gl,program),cachedAttributes=fetchAttributeLocations(gl,program)}__name(onFirstUse,"onFirstUse");let cachedUniforms;this.getUniforms=function(){return cachedUniforms===void 0&&onFirstUse(this),cachedUniforms};let cachedAttributes;this.getAttributes=function(){return cachedAttributes===void 0&&onFirstUse(this),cachedAttributes};let programReady=parameters.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return programReady===!1&&(programReady=gl.getProgramParameter(program,COMPLETION_STATUS_KHR)),programReady},this.destroy=function(){bindingStates.releaseStatesOfProgram(this),gl.deleteProgram(program),this.program=void 0},this.type=parameters.shaderType,this.name=parameters.shaderName,this.id=programIdCount++,this.cacheKey=cacheKey,this.usedTimes=1,this.program=program,this.vertexShader=glVertexShader,this.fragmentShader=glFragmentShader,this}__name(WebGLProgram,"WebGLProgram");let _id$1=0;class WebGLShaderCache{static{__name(this,"WebGLShaderCache")}constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(material){const vertexShader=material.vertexShader,fragmentShader=material.fragmentShader,vertexShaderStage=this._getShaderStage(vertexShader),fragmentShaderStage=this._getShaderStage(fragmentShader),materialShaders=this._getShaderCacheForMaterial(material);return materialShaders.has(vertexShaderStage)===!1&&(materialShaders.add(vertexShaderStage),vertexShaderStage.usedTimes++),materialShaders.has(fragmentShaderStage)===!1&&(materialShaders.add(fragmentShaderStage),fragmentShaderStage.usedTimes++),this}remove(material){const materialShaders=this.materialCache.get(material);for(const shaderStage of materialShaders)shaderStage.usedTimes--,shaderStage.usedTimes===0&&this.shaderCache.delete(shaderStage.code);return this.materialCache.delete(material),this}getVertexShaderID(material){return this._getShaderStage(material.vertexShader).id}getFragmentShaderID(material){return this._getShaderStage(material.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(material){const cache=this.materialCache;let set=cache.get(material);return set===void 0&&(set=new Set,cache.set(material,set)),set}_getShaderStage(code){const cache=this.shaderCache;let stage=cache.get(code);return stage===void 0&&(stage=new WebGLShaderStage(code),cache.set(code,stage)),stage}}class WebGLShaderStage{static{__name(this,"WebGLShaderStage")}constructor(code){this.id=_id$1++,this.code=code,this.usedTimes=0}}function WebGLPrograms(renderer,cubemaps,cubeuvmaps,extensions,capabilities,bindingStates,clipping){const _programLayers=new Layers,_customShaders=new WebGLShaderCache,_activeChannels=new Set,programs=[],logarithmicDepthBuffer=capabilities.logarithmicDepthBuffer,SUPPORTS_VERTEX_TEXTURES=capabilities.vertexTextures;let precision=capabilities.precision;const shaderIDs={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function getChannel(value){return _activeChannels.add(value),value===0?"uv":`uv${value}`}__name(getChannel,"getChannel");function getParameters(material,lights,shadows,scene,object){const fog=scene.fog,geometry=object.geometry,environment=material.isMeshStandardMaterial?scene.environment:null,envMap=(material.isMeshStandardMaterial?cubeuvmaps:cubemaps).get(material.envMap||environment),envMapCubeUVHeight=envMap&&envMap.mapping===CubeUVReflectionMapping?envMap.image.height:null,shaderID=shaderIDs[material.type];material.precision!==null&&(precision=capabilities.getMaxPrecision(material.precision),precision!==material.precision&&console.warn("THREE.WebGLProgram.getParameters:",material.precision,"not supported, using",precision,"instead."));const morphAttribute=geometry.morphAttributes.position||geometry.morphAttributes.normal||geometry.morphAttributes.color,morphTargetsCount=morphAttribute!==void 0?morphAttribute.length:0;let morphTextureStride=0;geometry.morphAttributes.position!==void 0&&(morphTextureStride=1),geometry.morphAttributes.normal!==void 0&&(morphTextureStride=2),geometry.morphAttributes.color!==void 0&&(morphTextureStride=3);let vertexShader,fragmentShader,customVertexShaderID,customFragmentShaderID;if(shaderID){const shader=ShaderLib[shaderID];vertexShader=shader.vertexShader,fragmentShader=shader.fragmentShader}else vertexShader=material.vertexShader,fragmentShader=material.fragmentShader,_customShaders.update(material),customVertexShaderID=_customShaders.getVertexShaderID(material),customFragmentShaderID=_customShaders.getFragmentShaderID(material);const currentRenderTarget=renderer.getRenderTarget(),reverseDepthBuffer=renderer.state.buffers.depth.getReversed(),IS_INSTANCEDMESH=object.isInstancedMesh===!0,IS_BATCHEDMESH=object.isBatchedMesh===!0,HAS_MAP=!!material.map,HAS_MATCAP=!!material.matcap,HAS_ENVMAP=!!envMap,HAS_AOMAP=!!material.aoMap,HAS_LIGHTMAP=!!material.lightMap,HAS_BUMPMAP=!!material.bumpMap,HAS_NORMALMAP=!!material.normalMap,HAS_DISPLACEMENTMAP=!!material.displacementMap,HAS_EMISSIVEMAP=!!material.emissiveMap,HAS_METALNESSMAP=!!material.metalnessMap,HAS_ROUGHNESSMAP=!!material.roughnessMap,HAS_ANISOTROPY=material.anisotropy>0,HAS_CLEARCOAT=material.clearcoat>0,HAS_DISPERSION=material.dispersion>0,HAS_IRIDESCENCE=material.iridescence>0,HAS_SHEEN=material.sheen>0,HAS_TRANSMISSION=material.transmission>0,HAS_ANISOTROPYMAP=HAS_ANISOTROPY&&!!material.anisotropyMap,HAS_CLEARCOATMAP=HAS_CLEARCOAT&&!!material.clearcoatMap,HAS_CLEARCOAT_NORMALMAP=HAS_CLEARCOAT&&!!material.clearcoatNormalMap,HAS_CLEARCOAT_ROUGHNESSMAP=HAS_CLEARCOAT&&!!material.clearcoatRoughnessMap,HAS_IRIDESCENCEMAP=HAS_IRIDESCENCE&&!!material.iridescenceMap,HAS_IRIDESCENCE_THICKNESSMAP=HAS_IRIDESCENCE&&!!material.iridescenceThicknessMap,HAS_SHEEN_COLORMAP=HAS_SHEEN&&!!material.sheenColorMap,HAS_SHEEN_ROUGHNESSMAP=HAS_SHEEN&&!!material.sheenRoughnessMap,HAS_SPECULARMAP=!!material.specularMap,HAS_SPECULAR_COLORMAP=!!material.specularColorMap,HAS_SPECULAR_INTENSITYMAP=!!material.specularIntensityMap,HAS_TRANSMISSIONMAP=HAS_TRANSMISSION&&!!material.transmissionMap,HAS_THICKNESSMAP=HAS_TRANSMISSION&&!!material.thicknessMap,HAS_GRADIENTMAP=!!material.gradientMap,HAS_ALPHAMAP=!!material.alphaMap,HAS_ALPHATEST=material.alphaTest>0,HAS_ALPHAHASH=!!material.alphaHash,HAS_EXTENSIONS=!!material.extensions;let toneMapping=NoToneMapping;material.toneMapped&&(currentRenderTarget===null||currentRenderTarget.isXRRenderTarget===!0)&&(toneMapping=renderer.toneMapping);const parameters={shaderID,shaderType:material.type,shaderName:material.name,vertexShader,fragmentShader,defines:material.defines,customVertexShaderID,customFragmentShaderID,isRawShaderMaterial:material.isRawShaderMaterial===!0,glslVersion:material.glslVersion,precision,batching:IS_BATCHEDMESH,batchingColor:IS_BATCHEDMESH&&object._colorsTexture!==null,instancing:IS_INSTANCEDMESH,instancingColor:IS_INSTANCEDMESH&&object.instanceColor!==null,instancingMorph:IS_INSTANCEDMESH&&object.morphTexture!==null,supportsVertexTextures:SUPPORTS_VERTEX_TEXTURES,outputColorSpace:currentRenderTarget===null?renderer.outputColorSpace:currentRenderTarget.isXRRenderTarget===!0?currentRenderTarget.texture.colorSpace:LinearSRGBColorSpace,alphaToCoverage:!!material.alphaToCoverage,map:HAS_MAP,matcap:HAS_MATCAP,envMap:HAS_ENVMAP,envMapMode:HAS_ENVMAP&&envMap.mapping,envMapCubeUVHeight,aoMap:HAS_AOMAP,lightMap:HAS_LIGHTMAP,bumpMap:HAS_BUMPMAP,normalMap:HAS_NORMALMAP,displacementMap:SUPPORTS_VERTEX_TEXTURES&&HAS_DISPLACEMENTMAP,emissiveMap:HAS_EMISSIVEMAP,normalMapObjectSpace:HAS_NORMALMAP&&material.normalMapType===ObjectSpaceNormalMap,normalMapTangentSpace:HAS_NORMALMAP&&material.normalMapType===TangentSpaceNormalMap,metalnessMap:HAS_METALNESSMAP,roughnessMap:HAS_ROUGHNESSMAP,anisotropy:HAS_ANISOTROPY,anisotropyMap:HAS_ANISOTROPYMAP,clearcoat:HAS_CLEARCOAT,clearcoatMap:HAS_CLEARCOATMAP,clearcoatNormalMap:HAS_CLEARCOAT_NORMALMAP,clearcoatRoughnessMap:HAS_CLEARCOAT_ROUGHNESSMAP,dispersion:HAS_DISPERSION,iridescence:HAS_IRIDESCENCE,iridescenceMap:HAS_IRIDESCENCEMAP,iridescenceThicknessMap:HAS_IRIDESCENCE_THICKNESSMAP,sheen:HAS_SHEEN,sheenColorMap:HAS_SHEEN_COLORMAP,sheenRoughnessMap:HAS_SHEEN_ROUGHNESSMAP,specularMap:HAS_SPECULARMAP,specularColorMap:HAS_SPECULAR_COLORMAP,specularIntensityMap:HAS_SPECULAR_INTENSITYMAP,transmission:HAS_TRANSMISSION,transmissionMap:HAS_TRANSMISSIONMAP,thicknessMap:HAS_THICKNESSMAP,gradientMap:HAS_GRADIENTMAP,opaque:material.transparent===!1&&material.blending===NormalBlending&&material.alphaToCoverage===!1,alphaMap:HAS_ALPHAMAP,alphaTest:HAS_ALPHATEST,alphaHash:HAS_ALPHAHASH,combine:material.combine,mapUv:HAS_MAP&&getChannel(material.map.channel),aoMapUv:HAS_AOMAP&&getChannel(material.aoMap.channel),lightMapUv:HAS_LIGHTMAP&&getChannel(material.lightMap.channel),bumpMapUv:HAS_BUMPMAP&&getChannel(material.bumpMap.channel),normalMapUv:HAS_NORMALMAP&&getChannel(material.normalMap.channel),displacementMapUv:HAS_DISPLACEMENTMAP&&getChannel(material.displacementMap.channel),emissiveMapUv:HAS_EMISSIVEMAP&&getChannel(material.emissiveMap.channel),metalnessMapUv:HAS_METALNESSMAP&&getChannel(material.metalnessMap.channel),roughnessMapUv:HAS_ROUGHNESSMAP&&getChannel(material.roughnessMap.channel),anisotropyMapUv:HAS_ANISOTROPYMAP&&getChannel(material.anisotropyMap.channel),clearcoatMapUv:HAS_CLEARCOATMAP&&getChannel(material.clearcoatMap.channel),clearcoatNormalMapUv:HAS_CLEARCOAT_NORMALMAP&&getChannel(material.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:HAS_CLEARCOAT_ROUGHNESSMAP&&getChannel(material.clearcoatRoughnessMap.channel),iridescenceMapUv:HAS_IRIDESCENCEMAP&&getChannel(material.iridescenceMap.channel),iridescenceThicknessMapUv:HAS_IRIDESCENCE_THICKNESSMAP&&getChannel(material.iridescenceThicknessMap.channel),sheenColorMapUv:HAS_SHEEN_COLORMAP&&getChannel(material.sheenColorMap.channel),sheenRoughnessMapUv:HAS_SHEEN_ROUGHNESSMAP&&getChannel(material.sheenRoughnessMap.channel),specularMapUv:HAS_SPECULARMAP&&getChannel(material.specularMap.channel),specularColorMapUv:HAS_SPECULAR_COLORMAP&&getChannel(material.specularColorMap.channel),specularIntensityMapUv:HAS_SPECULAR_INTENSITYMAP&&getChannel(material.specularIntensityMap.channel),transmissionMapUv:HAS_TRANSMISSIONMAP&&getChannel(material.transmissionMap.channel),thicknessMapUv:HAS_THICKNESSMAP&&getChannel(material.thicknessMap.channel),alphaMapUv:HAS_ALPHAMAP&&getChannel(material.alphaMap.channel),vertexTangents:!!geometry.attributes.tangent&&(HAS_NORMALMAP||HAS_ANISOTROPY),vertexColors:material.vertexColors,vertexAlphas:material.vertexColors===!0&&!!geometry.attributes.color&&geometry.attributes.color.itemSize===4,pointsUvs:object.isPoints===!0&&!!geometry.attributes.uv&&(HAS_MAP||HAS_ALPHAMAP),fog:!!fog,useFog:material.fog===!0,fogExp2:!!fog&&fog.isFogExp2,flatShading:material.flatShading===!0,sizeAttenuation:material.sizeAttenuation===!0,logarithmicDepthBuffer,reverseDepthBuffer,skinning:object.isSkinnedMesh===!0,morphTargets:geometry.morphAttributes.position!==void 0,morphNormals:geometry.morphAttributes.normal!==void 0,morphColors:geometry.morphAttributes.color!==void 0,morphTargetsCount,morphTextureStride,numDirLights:lights.directional.length,numPointLights:lights.point.length,numSpotLights:lights.spot.length,numSpotLightMaps:lights.spotLightMap.length,numRectAreaLights:lights.rectArea.length,numHemiLights:lights.hemi.length,numDirLightShadows:lights.directionalShadowMap.length,numPointLightShadows:lights.pointShadowMap.length,numSpotLightShadows:lights.spotShadowMap.length,numSpotLightShadowsWithMaps:lights.numSpotLightShadowsWithMaps,numLightProbes:lights.numLightProbes,numClippingPlanes:clipping.numPlanes,numClipIntersection:clipping.numIntersection,dithering:material.dithering,shadowMapEnabled:renderer.shadowMap.enabled&&shadows.length>0,shadowMapType:renderer.shadowMap.type,toneMapping,decodeVideoTexture:HAS_MAP&&material.map.isVideoTexture===!0&&ColorManagement.getTransfer(material.map.colorSpace)===SRGBTransfer,decodeVideoTextureEmissive:HAS_EMISSIVEMAP&&material.emissiveMap.isVideoTexture===!0&&ColorManagement.getTransfer(material.emissiveMap.colorSpace)===SRGBTransfer,premultipliedAlpha:material.premultipliedAlpha,doubleSided:material.side===DoubleSide,flipSided:material.side===BackSide,useDepthPacking:material.depthPacking>=0,depthPacking:material.depthPacking||0,index0AttributeName:material.index0AttributeName,extensionClipCullDistance:HAS_EXTENSIONS&&material.extensions.clipCullDistance===!0&&extensions.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(HAS_EXTENSIONS&&material.extensions.multiDraw===!0||IS_BATCHEDMESH)&&extensions.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:extensions.has("KHR_parallel_shader_compile"),customProgramCacheKey:material.customProgramCacheKey()};return parameters.vertexUv1s=_activeChannels.has(1),parameters.vertexUv2s=_activeChannels.has(2),parameters.vertexUv3s=_activeChannels.has(3),_activeChannels.clear(),parameters}__name(getParameters,"getParameters");function getProgramCacheKey(parameters){const array=[];if(parameters.shaderID?array.push(parameters.shaderID):(array.push(parameters.customVertexShaderID),array.push(parameters.customFragmentShaderID)),parameters.defines!==void 0)for(const name in parameters.defines)array.push(name),array.push(parameters.defines[name]);return parameters.isRawShaderMaterial===!1&&(getProgramCacheKeyParameters(array,parameters),getProgramCacheKeyBooleans(array,parameters),array.push(renderer.outputColorSpace)),array.push(parameters.customProgramCacheKey),array.join()}__name(getProgramCacheKey,"getProgramCacheKey");function getProgramCacheKeyParameters(array,parameters){array.push(parameters.precision),array.push(parameters.outputColorSpace),array.push(parameters.envMapMode),array.push(parameters.envMapCubeUVHeight),array.push(parameters.mapUv),array.push(parameters.alphaMapUv),array.push(parameters.lightMapUv),array.push(parameters.aoMapUv),array.push(parameters.bumpMapUv),array.push(parameters.normalMapUv),array.push(parameters.displacementMapUv),array.push(parameters.emissiveMapUv),array.push(parameters.metalnessMapUv),array.push(parameters.roughnessMapUv),array.push(parameters.anisotropyMapUv),array.push(parameters.clearcoatMapUv),array.push(parameters.clearcoatNormalMapUv),array.push(parameters.clearcoatRoughnessMapUv),array.push(parameters.iridescenceMapUv),array.push(parameters.iridescenceThicknessMapUv),array.push(parameters.sheenColorMapUv),array.push(parameters.sheenRoughnessMapUv),array.push(parameters.specularMapUv),array.push(parameters.specularColorMapUv),array.push(parameters.specularIntensityMapUv),array.push(parameters.transmissionMapUv),array.push(parameters.thicknessMapUv),array.push(parameters.combine),array.push(parameters.fogExp2),array.push(parameters.sizeAttenuation),array.push(parameters.morphTargetsCount),array.push(parameters.morphAttributeCount),array.push(parameters.numDirLights),array.push(parameters.numPointLights),array.push(parameters.numSpotLights),array.push(parameters.numSpotLightMaps),array.push(parameters.numHemiLights),array.push(parameters.numRectAreaLights),array.push(parameters.numDirLightShadows),array.push(parameters.numPointLightShadows),array.push(parameters.numSpotLightShadows),array.push(parameters.numSpotLightShadowsWithMaps),array.push(parameters.numLightProbes),array.push(parameters.shadowMapType),array.push(parameters.toneMapping),array.push(parameters.numClippingPlanes),array.push(parameters.numClipIntersection),array.push(parameters.depthPacking)}__name(getProgramCacheKeyParameters,"getProgramCacheKeyParameters");function getProgramCacheKeyBooleans(array,parameters){_programLayers.disableAll(),parameters.supportsVertexTextures&&_programLayers.enable(0),parameters.instancing&&_programLayers.enable(1),parameters.instancingColor&&_programLayers.enable(2),parameters.instancingMorph&&_programLayers.enable(3),parameters.matcap&&_programLayers.enable(4),parameters.envMap&&_programLayers.enable(5),parameters.normalMapObjectSpace&&_programLayers.enable(6),parameters.normalMapTangentSpace&&_programLayers.enable(7),parameters.clearcoat&&_programLayers.enable(8),parameters.iridescence&&_programLayers.enable(9),parameters.alphaTest&&_programLayers.enable(10),parameters.vertexColors&&_programLayers.enable(11),parameters.vertexAlphas&&_programLayers.enable(12),parameters.vertexUv1s&&_programLayers.enable(13),parameters.vertexUv2s&&_programLayers.enable(14),parameters.vertexUv3s&&_programLayers.enable(15),parameters.vertexTangents&&_programLayers.enable(16),parameters.anisotropy&&_programLayers.enable(17),parameters.alphaHash&&_programLayers.enable(18),parameters.batching&&_programLayers.enable(19),parameters.dispersion&&_programLayers.enable(20),parameters.batchingColor&&_programLayers.enable(21),array.push(_programLayers.mask),_programLayers.disableAll(),parameters.fog&&_programLayers.enable(0),parameters.useFog&&_programLayers.enable(1),parameters.flatShading&&_programLayers.enable(2),parameters.logarithmicDepthBuffer&&_programLayers.enable(3),parameters.reverseDepthBuffer&&_programLayers.enable(4),parameters.skinning&&_programLayers.enable(5),parameters.morphTargets&&_programLayers.enable(6),parameters.morphNormals&&_programLayers.enable(7),parameters.morphColors&&_programLayers.enable(8),parameters.premultipliedAlpha&&_programLayers.enable(9),parameters.shadowMapEnabled&&_programLayers.enable(10),parameters.doubleSided&&_programLayers.enable(11),parameters.flipSided&&_programLayers.enable(12),parameters.useDepthPacking&&_programLayers.enable(13),parameters.dithering&&_programLayers.enable(14),parameters.transmission&&_programLayers.enable(15),parameters.sheen&&_programLayers.enable(16),parameters.opaque&&_programLayers.enable(17),parameters.pointsUvs&&_programLayers.enable(18),parameters.decodeVideoTexture&&_programLayers.enable(19),parameters.decodeVideoTextureEmissive&&_programLayers.enable(20),parameters.alphaToCoverage&&_programLayers.enable(21),array.push(_programLayers.mask)}__name(getProgramCacheKeyBooleans,"getProgramCacheKeyBooleans");function getUniforms(material){const shaderID=shaderIDs[material.type];let uniforms;if(shaderID){const shader=ShaderLib[shaderID];uniforms=UniformsUtils.clone(shader.uniforms)}else uniforms=material.uniforms;return uniforms}__name(getUniforms,"getUniforms");function acquireProgram(parameters,cacheKey){let program;for(let p=0,pl=programs.length;p<pl;p++){const preexistingProgram=programs[p];if(preexistingProgram.cacheKey===cacheKey){program=preexistingProgram,++program.usedTimes;break}}return program===void 0&&(program=new WebGLProgram(renderer,cacheKey,parameters,bindingStates),programs.push(program)),program}__name(acquireProgram,"acquireProgram");function releaseProgram(program){if(--program.usedTimes===0){const i=programs.indexOf(program);programs[i]=programs[programs.length-1],programs.pop(),program.destroy()}}__name(releaseProgram,"releaseProgram");function releaseShaderCache(material){_customShaders.remove(material)}__name(releaseShaderCache,"releaseShaderCache");function dispose(){_customShaders.dispose()}return __name(dispose,"dispose"),{getParameters,getProgramCacheKey,getUniforms,acquireProgram,releaseProgram,releaseShaderCache,programs,dispose}}__name(WebGLPrograms,"WebGLPrograms");function WebGLProperties(){let properties=new WeakMap;function has(object){return properties.has(object)}__name(has,"has");function get(object){let map=properties.get(object);return map===void 0&&(map={},properties.set(object,map)),map}__name(get,"get");function remove(object){properties.delete(object)}__name(remove,"remove");function update(object,key,value){properties.get(object)[key]=value}__name(update,"update");function dispose(){properties=new WeakMap}return __name(dispose,"dispose"),{has,get,remove,update,dispose}}__name(WebGLProperties,"WebGLProperties");function painterSortStable(a,b){return a.groupOrder!==b.groupOrder?a.groupOrder-b.groupOrder:a.renderOrder!==b.renderOrder?a.renderOrder-b.renderOrder:a.material.id!==b.material.id?a.material.id-b.material.id:a.z!==b.z?a.z-b.z:a.id-b.id}__name(painterSortStable,"painterSortStable");function reversePainterSortStable(a,b){return a.groupOrder!==b.groupOrder?a.groupOrder-b.groupOrder:a.renderOrder!==b.renderOrder?a.renderOrder-b.renderOrder:a.z!==b.z?b.z-a.z:a.id-b.id}__name(reversePainterSortStable,"reversePainterSortStable");function WebGLRenderList(){const renderItems=[];let renderItemsIndex=0;const opaque=[],transmissive=[],transparent=[];function init(){renderItemsIndex=0,opaque.length=0,transmissive.length=0,transparent.length=0}__name(init,"init");function getNextRenderItem(object,geometry,material,groupOrder,z,group){let renderItem=renderItems[renderItemsIndex];return renderItem===void 0?(renderItem={id:object.id,object,geometry,material,groupOrder,renderOrder:object.renderOrder,z,group},renderItems[renderItemsIndex]=renderItem):(renderItem.id=object.id,renderItem.object=object,renderItem.geometry=geometry,renderItem.material=material,renderItem.groupOrder=groupOrder,renderItem.renderOrder=object.renderOrder,renderItem.z=z,renderItem.group=group),renderItemsIndex++,renderItem}__name(getNextRenderItem,"getNextRenderItem");function push(object,geometry,material,groupOrder,z,group){const renderItem=getNextRenderItem(object,geometry,material,groupOrder,z,group);material.transmission>0?transmissive.push(renderItem):material.transparent===!0?transparent.push(renderItem):opaque.push(renderItem)}__name(push,"push");function unshift(object,geometry,material,groupOrder,z,group){const renderItem=getNextRenderItem(object,geometry,material,groupOrder,z,group);material.transmission>0?transmissive.unshift(renderItem):material.transparent===!0?transparent.unshift(renderItem):opaque.unshift(renderItem)}__name(unshift,"unshift");function sort(customOpaqueSort,customTransparentSort){opaque.length>1&&opaque.sort(customOpaqueSort||painterSortStable),transmissive.length>1&&transmissive.sort(customTransparentSort||reversePainterSortStable),transparent.length>1&&transparent.sort(customTransparentSort||reversePainterSortStable)}__name(sort,"sort");function finish(){for(let i=renderItemsIndex,il=renderItems.length;i<il;i++){const renderItem=renderItems[i];if(renderItem.id===null)break;renderItem.id=null,renderItem.object=null,renderItem.geometry=null,renderItem.material=null,renderItem.group=null}}return __name(finish,"finish"),{opaque,transmissive,transparent,init,push,unshift,finish,sort}}__name(WebGLRenderList,"WebGLRenderList");function WebGLRenderLists(){let lists=new WeakMap;function get(scene,renderCallDepth){const listArray=lists.get(scene);let list;return listArray===void 0?(list=new WebGLRenderList,lists.set(scene,[list])):renderCallDepth>=listArray.length?(list=new WebGLRenderList,listArray.push(list)):list=listArray[renderCallDepth],list}__name(get,"get");function dispose(){lists=new WeakMap}return __name(dispose,"dispose"),{get,dispose}}__name(WebGLRenderLists,"WebGLRenderLists");function UniformsCache(){const lights={};return{get:__name(function(light){if(lights[light.id]!==void 0)return lights[light.id];let uniforms;switch(light.type){case"DirectionalLight":uniforms={direction:new Vector3,color:new Color};break;case"SpotLight":uniforms={position:new Vector3,direction:new Vector3,color:new Color,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":uniforms={position:new Vector3,color:new Color,distance:0,decay:0};break;case"HemisphereLight":uniforms={direction:new Vector3,skyColor:new Color,groundColor:new Color};break;case"RectAreaLight":uniforms={color:new Color,position:new Vector3,halfWidth:new Vector3,halfHeight:new Vector3};break}return lights[light.id]=uniforms,uniforms},"get")}}__name(UniformsCache,"UniformsCache");function ShadowUniformsCache(){const lights={};return{get:__name(function(light){if(lights[light.id]!==void 0)return lights[light.id];let uniforms;switch(light.type){case"DirectionalLight":uniforms={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Vector2};break;case"SpotLight":uniforms={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Vector2};break;case"PointLight":uniforms={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Vector2,shadowCameraNear:1,shadowCameraFar:1e3};break}return lights[light.id]=uniforms,uniforms},"get")}}__name(ShadowUniformsCache,"ShadowUniformsCache");let nextVersion=0;function shadowCastingAndTexturingLightsFirst(lightA,lightB){return(lightB.castShadow?2:0)-(lightA.castShadow?2:0)+(lightB.map?1:0)-(lightA.map?1:0)}__name(shadowCastingAndTexturingLightsFirst,"shadowCastingAndTexturingLightsFirst");function WebGLLights(extensions){const cache=new UniformsCache,shadowCache=ShadowUniformsCache(),state={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let i=0;i<9;i++)state.probe.push(new Vector3);const vector3=new Vector3,matrix4=new Matrix4,matrix42=new Matrix4;function setup(lights){let r=0,g=0,b=0;for(let i=0;i<9;i++)state.probe[i].set(0,0,0);let directionalLength=0,pointLength=0,spotLength=0,rectAreaLength=0,hemiLength=0,numDirectionalShadows=0,numPointShadows=0,numSpotShadows=0,numSpotMaps=0,numSpotShadowsWithMaps=0,numLightProbes=0;lights.sort(shadowCastingAndTexturingLightsFirst);for(let i=0,l=lights.length;i<l;i++){const light=lights[i],color=light.color,intensity=light.intensity,distance=light.distance,shadowMap=light.shadow&&light.shadow.map?light.shadow.map.texture:null;if(light.isAmbientLight)r+=color.r*intensity,g+=color.g*intensity,b+=color.b*intensity;else if(light.isLightProbe){for(let j=0;j<9;j++)state.probe[j].addScaledVector(light.sh.coefficients[j],intensity);numLightProbes++}else if(light.isDirectionalLight){const uniforms=cache.get(light);if(uniforms.color.copy(light.color).multiplyScalar(light.intensity),light.castShadow){const shadow=light.shadow,shadowUniforms=shadowCache.get(light);shadowUniforms.shadowIntensity=shadow.intensity,shadowUniforms.shadowBias=shadow.bias,shadowUniforms.shadowNormalBias=shadow.normalBias,shadowUniforms.shadowRadius=shadow.radius,shadowUniforms.shadowMapSize=shadow.mapSize,state.directionalShadow[directionalLength]=shadowUniforms,state.directionalShadowMap[directionalLength]=shadowMap,state.directionalShadowMatrix[directionalLength]=light.shadow.matrix,numDirectionalShadows++}state.directional[directionalLength]=uniforms,directionalLength++}else if(light.isSpotLight){const uniforms=cache.get(light);uniforms.position.setFromMatrixPosition(light.matrixWorld),uniforms.color.copy(color).multiplyScalar(intensity),uniforms.distance=distance,uniforms.coneCos=Math.cos(light.angle),uniforms.penumbraCos=Math.cos(light.angle*(1-light.penumbra)),uniforms.decay=light.decay,state.spot[spotLength]=uniforms;const shadow=light.shadow;if(light.map&&(state.spotLightMap[numSpotMaps]=light.map,numSpotMaps++,shadow.updateMatrices(light),light.castShadow&&numSpotShadowsWithMaps++),state.spotLightMatrix[spotLength]=shadow.matrix,light.castShadow){const shadowUniforms=shadowCache.get(light);shadowUniforms.shadowIntensity=shadow.intensity,shadowUniforms.shadowBias=shadow.bias,shadowUniforms.shadowNormalBias=shadow.normalBias,shadowUniforms.shadowRadius=shadow.radius,shadowUniforms.shadowMapSize=shadow.mapSize,state.spotShadow[spotLength]=shadowUniforms,state.spotShadowMap[spotLength]=shadowMap,numSpotShadows++}spotLength++}else if(light.isRectAreaLight){const uniforms=cache.get(light);uniforms.color.copy(color).multiplyScalar(intensity),uniforms.halfWidth.set(light.width*.5,0,0),uniforms.halfHeight.set(0,light.height*.5,0),state.rectArea[rectAreaLength]=uniforms,rectAreaLength++}else if(light.isPointLight){const uniforms=cache.get(light);if(uniforms.color.copy(light.color).multiplyScalar(light.intensity),uniforms.distance=light.distance,uniforms.decay=light.decay,light.castShadow){const shadow=light.shadow,shadowUniforms=shadowCache.get(light);shadowUniforms.shadowIntensity=shadow.intensity,shadowUniforms.shadowBias=shadow.bias,shadowUniforms.shadowNormalBias=shadow.normalBias,shadowUniforms.shadowRadius=shadow.radius,shadowUniforms.shadowMapSize=shadow.mapSize,shadowUniforms.shadowCameraNear=shadow.camera.near,shadowUniforms.shadowCameraFar=shadow.camera.far,state.pointShadow[pointLength]=shadowUniforms,state.pointShadowMap[pointLength]=shadowMap,state.pointShadowMatrix[pointLength]=light.shadow.matrix,numPointShadows++}state.point[pointLength]=uniforms,pointLength++}else if(light.isHemisphereLight){const uniforms=cache.get(light);uniforms.skyColor.copy(light.color).multiplyScalar(intensity),uniforms.groundColor.copy(light.groundColor).multiplyScalar(intensity),state.hemi[hemiLength]=uniforms,hemiLength++}}rectAreaLength>0&&(extensions.has("OES_texture_float_linear")===!0?(state.rectAreaLTC1=UniformsLib.LTC_FLOAT_1,state.rectAreaLTC2=UniformsLib.LTC_FLOAT_2):(state.rectAreaLTC1=UniformsLib.LTC_HALF_1,state.rectAreaLTC2=UniformsLib.LTC_HALF_2)),state.ambient[0]=r,state.ambient[1]=g,state.ambient[2]=b;const hash=state.hash;(hash.directionalLength!==directionalLength||hash.pointLength!==pointLength||hash.spotLength!==spotLength||hash.rectAreaLength!==rectAreaLength||hash.hemiLength!==hemiLength||hash.numDirectionalShadows!==numDirectionalShadows||hash.numPointShadows!==numPointShadows||hash.numSpotShadows!==numSpotShadows||hash.numSpotMaps!==numSpotMaps||hash.numLightProbes!==numLightProbes)&&(state.directional.length=directionalLength,state.spot.length=spotLength,state.rectArea.length=rectAreaLength,state.point.length=pointLength,state.hemi.length=hemiLength,state.directionalShadow.length=numDirectionalShadows,state.directionalShadowMap.length=numDirectionalShadows,state.pointShadow.length=numPointShadows,state.pointShadowMap.length=numPointShadows,state.spotShadow.length=numSpotShadows,state.spotShadowMap.length=numSpotShadows,state.directionalShadowMatrix.length=numDirectionalShadows,state.pointShadowMatrix.length=numPointShadows,state.spotLightMatrix.length=numSpotShadows+numSpotMaps-numSpotShadowsWithMaps,state.spotLightMap.length=numSpotMaps,state.numSpotLightShadowsWithMaps=numSpotShadowsWithMaps,state.numLightProbes=numLightProbes,hash.directionalLength=directionalLength,hash.pointLength=pointLength,hash.spotLength=spotLength,hash.rectAreaLength=rectAreaLength,hash.hemiLength=hemiLength,hash.numDirectionalShadows=numDirectionalShadows,hash.numPointShadows=numPointShadows,hash.numSpotShadows=numSpotShadows,hash.numSpotMaps=numSpotMaps,hash.numLightProbes=numLightProbes,state.version=nextVersion++)}__name(setup,"setup");function setupView(lights,camera){let directionalLength=0,pointLength=0,spotLength=0,rectAreaLength=0,hemiLength=0;const viewMatrix=camera.matrixWorldInverse;for(let i=0,l=lights.length;i<l;i++){const light=lights[i];if(light.isDirectionalLight){const uniforms=state.directional[directionalLength];uniforms.direction.setFromMatrixPosition(light.matrixWorld),vector3.setFromMatrixPosition(light.target.matrixWorld),uniforms.direction.sub(vector3),uniforms.direction.transformDirection(viewMatrix),directionalLength++}else if(light.isSpotLight){const uniforms=state.spot[spotLength];uniforms.position.setFromMatrixPosition(light.matrixWorld),uniforms.position.applyMatrix4(viewMatrix),uniforms.direction.setFromMatrixPosition(light.matrixWorld),vector3.setFromMatrixPosition(light.target.matrixWorld),uniforms.direction.sub(vector3),uniforms.direction.transformDirection(viewMatrix),spotLength++}else if(light.isRectAreaLight){const uniforms=state.rectArea[rectAreaLength];uniforms.position.setFromMatrixPosition(light.matrixWorld),uniforms.position.applyMatrix4(viewMatrix),matrix42.identity(),matrix4.copy(light.matrixWorld),matrix4.premultiply(viewMatrix),matrix42.extractRotation(matrix4),uniforms.halfWidth.set(light.width*.5,0,0),uniforms.halfHeight.set(0,light.height*.5,0),uniforms.halfWidth.applyMatrix4(matrix42),uniforms.halfHeight.applyMatrix4(matrix42),rectAreaLength++}else if(light.isPointLight){const uniforms=state.point[pointLength];uniforms.position.setFromMatrixPosition(light.matrixWorld),uniforms.position.applyMatrix4(viewMatrix),pointLength++}else if(light.isHemisphereLight){const uniforms=state.hemi[hemiLength];uniforms.direction.setFromMatrixPosition(light.matrixWorld),uniforms.direction.transformDirection(viewMatrix),hemiLength++}}}return __name(setupView,"setupView"),{setup,setupView,state}}__name(WebGLLights,"WebGLLights");function WebGLRenderState(extensions){const lights=new WebGLLights(extensions),lightsArray=[],shadowsArray=[];function init(camera){state.camera=camera,lightsArray.length=0,shadowsArray.length=0}__name(init,"init");function pushLight(light){lightsArray.push(light)}__name(pushLight,"pushLight");function pushShadow(shadowLight){shadowsArray.push(shadowLight)}__name(pushShadow,"pushShadow");function setupLights(){lights.setup(lightsArray)}__name(setupLights,"setupLights");function setupLightsView(camera){lights.setupView(lightsArray,camera)}__name(setupLightsView,"setupLightsView");const state={lightsArray,shadowsArray,camera:null,lights,transmissionRenderTarget:{}};return{init,state,setupLights,setupLightsView,pushLight,pushShadow}}__name(WebGLRenderState,"WebGLRenderState");function WebGLRenderStates(extensions){let renderStates=new WeakMap;function get(scene,renderCallDepth=0){const renderStateArray=renderStates.get(scene);let renderState;return renderStateArray===void 0?(renderState=new WebGLRenderState(extensions),renderStates.set(scene,[renderState])):renderCallDepth>=renderStateArray.length?(renderState=new WebGLRenderState(extensions),renderStateArray.push(renderState)):renderState=renderStateArray[renderCallDepth],renderState}__name(get,"get");function dispose(){renderStates=new WeakMap}return __name(dispose,"dispose"),{get,dispose}}__name(WebGLRenderStates,"WebGLRenderStates");class MeshDepthMaterial extends Material{static{__name(this,"MeshDepthMaterial")}static get type(){return"MeshDepthMaterial"}constructor(parameters){super(),this.isMeshDepthMaterial=!0,this.depthPacking=BasicDepthPacking,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(parameters)}copy(source){return super.copy(source),this.depthPacking=source.depthPacking,this.map=source.map,this.alphaMap=source.alphaMap,this.displacementMap=source.displacementMap,this.displacementScale=source.displacementScale,this.displacementBias=source.displacementBias,this.wireframe=source.wireframe,this.wireframeLinewidth=source.wireframeLinewidth,this}}class MeshDistanceMaterial extends Material{static{__name(this,"MeshDistanceMaterial")}static get type(){return"MeshDistanceMaterial"}constructor(parameters){super(),this.isMeshDistanceMaterial=!0,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(parameters)}copy(source){return super.copy(source),this.map=source.map,this.alphaMap=source.alphaMap,this.displacementMap=source.displacementMap,this.displacementScale=source.displacementScale,this.displacementBias=source.displacementBias,this}}const vertex=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,fragment=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function WebGLShadowMap(renderer,objects,capabilities){let _frustum=new Frustum;const _shadowMapSize=new Vector2,_viewportSize=new Vector2,_viewport=new Vector4,_depthMaterial=new MeshDepthMaterial({depthPacking:RGBADepthPacking}),_distanceMaterial=new MeshDistanceMaterial,_materialCache={},_maxTextureSize=capabilities.maxTextureSize,shadowSide={[FrontSide]:BackSide,[BackSide]:FrontSide,[DoubleSide]:DoubleSide},shadowMaterialVertical=new ShaderMaterial({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Vector2},radius:{value:4}},vertexShader:vertex,fragmentShader:fragment}),shadowMaterialHorizontal=shadowMaterialVertical.clone();shadowMaterialHorizontal.defines.HORIZONTAL_PASS=1;const fullScreenTri=new BufferGeometry;fullScreenTri.setAttribute("position",new BufferAttribute(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const fullScreenMesh=new Mesh(fullScreenTri,shadowMaterialVertical),scope=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=PCFShadowMap;let _previousType=this.type;this.render=function(lights,scene,camera){if(scope.enabled===!1||scope.autoUpdate===!1&&scope.needsUpdate===!1||lights.length===0)return;const currentRenderTarget=renderer.getRenderTarget(),activeCubeFace=renderer.getActiveCubeFace(),activeMipmapLevel=renderer.getActiveMipmapLevel(),_state=renderer.state;_state.setBlending(NoBlending),_state.buffers.color.setClear(1,1,1,1),_state.buffers.depth.setTest(!0),_state.setScissorTest(!1);const toVSM=_previousType!==VSMShadowMap&&this.type===VSMShadowMap,fromVSM=_previousType===VSMShadowMap&&this.type!==VSMShadowMap;for(let i=0,il=lights.length;i<il;i++){const light=lights[i],shadow=light.shadow;if(shadow===void 0){console.warn("THREE.WebGLShadowMap:",light,"has no shadow.");continue}if(shadow.autoUpdate===!1&&shadow.needsUpdate===!1)continue;_shadowMapSize.copy(shadow.mapSize);const shadowFrameExtents=shadow.getFrameExtents();if(_shadowMapSize.multiply(shadowFrameExtents),_viewportSize.copy(shadow.mapSize),(_shadowMapSize.x>_maxTextureSize||_shadowMapSize.y>_maxTextureSize)&&(_shadowMapSize.x>_maxTextureSize&&(_viewportSize.x=Math.floor(_maxTextureSize/shadowFrameExtents.x),_shadowMapSize.x=_viewportSize.x*shadowFrameExtents.x,shadow.mapSize.x=_viewportSize.x),_shadowMapSize.y>_maxTextureSize&&(_viewportSize.y=Math.floor(_maxTextureSize/shadowFrameExtents.y),_shadowMapSize.y=_viewportSize.y*shadowFrameExtents.y,shadow.mapSize.y=_viewportSize.y)),shadow.map===null||toVSM===!0||fromVSM===!0){const pars=this.type!==VSMShadowMap?{minFilter:NearestFilter,magFilter:NearestFilter}:{};shadow.map!==null&&shadow.map.dispose(),shadow.map=new WebGLRenderTarget(_shadowMapSize.x,_shadowMapSize.y,pars),shadow.map.texture.name=light.name+".shadowMap",shadow.camera.updateProjectionMatrix()}renderer.setRenderTarget(shadow.map),renderer.clear();const viewportCount=shadow.getViewportCount();for(let vp=0;vp<viewportCount;vp++){const viewport=shadow.getViewport(vp);_viewport.set(_viewportSize.x*viewport.x,_viewportSize.y*viewport.y,_viewportSize.x*viewport.z,_viewportSize.y*viewport.w),_state.viewport(_viewport),shadow.updateMatrices(light,vp),_frustum=shadow.getFrustum(),renderObject(scene,camera,shadow.camera,light,this.type)}shadow.isPointLightShadow!==!0&&this.type===VSMShadowMap&&VSMPass(shadow,camera),shadow.needsUpdate=!1}_previousType=this.type,scope.needsUpdate=!1,renderer.setRenderTarget(currentRenderTarget,activeCubeFace,activeMipmapLevel)};function VSMPass(shadow,camera){const geometry=objects.update(fullScreenMesh);shadowMaterialVertical.defines.VSM_SAMPLES!==shadow.blurSamples&&(shadowMaterialVertical.defines.VSM_SAMPLES=shadow.blurSamples,shadowMaterialHorizontal.defines.VSM_SAMPLES=shadow.blurSamples,shadowMaterialVertical.needsUpdate=!0,shadowMaterialHorizontal.needsUpdate=!0),shadow.mapPass===null&&(shadow.mapPass=new WebGLRenderTarget(_shadowMapSize.x,_shadowMapSize.y)),shadowMaterialVertical.uniforms.shadow_pass.value=shadow.map.texture,shadowMaterialVertical.uniforms.resolution.value=shadow.mapSize,shadowMaterialVertical.uniforms.radius.value=shadow.radius,renderer.setRenderTarget(shadow.mapPass),renderer.clear(),renderer.renderBufferDirect(camera,null,geometry,shadowMaterialVertical,fullScreenMesh,null),shadowMaterialHorizontal.uniforms.shadow_pass.value=shadow.mapPass.texture,shadowMaterialHorizontal.uniforms.resolution.value=shadow.mapSize,shadowMaterialHorizontal.uniforms.radius.value=shadow.radius,renderer.setRenderTarget(shadow.map),renderer.clear(),renderer.renderBufferDirect(camera,null,geometry,shadowMaterialHorizontal,fullScreenMesh,null)}__name(VSMPass,"VSMPass");function getDepthMaterial(object,material,light,type){let result=null;const customMaterial=light.isPointLight===!0?object.customDistanceMaterial:object.customDepthMaterial;if(customMaterial!==void 0)result=customMaterial;else if(result=light.isPointLight===!0?_distanceMaterial:_depthMaterial,renderer.localClippingEnabled&&material.clipShadows===!0&&Array.isArray(material.clippingPlanes)&&material.clippingPlanes.length!==0||material.displacementMap&&material.displacementScale!==0||material.alphaMap&&material.alphaTest>0||material.map&&material.alphaTest>0){const keyA=result.uuid,keyB=material.uuid;let materialsForVariant=_materialCache[keyA];materialsForVariant===void 0&&(materialsForVariant={},_materialCache[keyA]=materialsForVariant);let cachedMaterial=materialsForVariant[keyB];cachedMaterial===void 0&&(cachedMaterial=result.clone(),materialsForVariant[keyB]=cachedMaterial,material.addEventListener("dispose",onMaterialDispose)),result=cachedMaterial}if(result.visible=material.visible,result.wireframe=material.wireframe,type===VSMShadowMap?result.side=material.shadowSide!==null?material.shadowSide:material.side:result.side=material.shadowSide!==null?material.shadowSide:shadowSide[material.side],result.alphaMap=material.alphaMap,result.alphaTest=material.alphaTest,result.map=material.map,result.clipShadows=material.clipShadows,result.clippingPlanes=material.clippingPlanes,result.clipIntersection=material.clipIntersection,result.displacementMap=material.displacementMap,result.displacementScale=material.displacementScale,result.displacementBias=material.displacementBias,result.wireframeLinewidth=material.wireframeLinewidth,result.linewidth=material.linewidth,light.isPointLight===!0&&result.isMeshDistanceMaterial===!0){const materialProperties=renderer.properties.get(result);materialProperties.light=light}return result}__name(getDepthMaterial,"getDepthMaterial");function renderObject(object,camera,shadowCamera,light,type){if(object.visible===!1)return;if(object.layers.test(camera.layers)&&(object.isMesh||object.isLine||object.isPoints)&&(object.castShadow||object.receiveShadow&&type===VSMShadowMap)&&(!object.frustumCulled||_frustum.intersectsObject(object))){object.modelViewMatrix.multiplyMatrices(shadowCamera.matrixWorldInverse,object.matrixWorld);const geometry=objects.update(object),material=object.material;if(Array.isArray(material)){const groups=geometry.groups;for(let k=0,kl=groups.length;k<kl;k++){const group=groups[k],groupMaterial=material[group.materialIndex];if(groupMaterial&&groupMaterial.visible){const depthMaterial=getDepthMaterial(object,groupMaterial,light,type);object.onBeforeShadow(renderer,object,camera,shadowCamera,geometry,depthMaterial,group),renderer.renderBufferDirect(shadowCamera,null,geometry,depthMaterial,object,group),object.onAfterShadow(renderer,object,camera,shadowCamera,geometry,depthMaterial,group)}}}else if(material.visible){const depthMaterial=getDepthMaterial(object,material,light,type);object.onBeforeShadow(renderer,object,camera,shadowCamera,geometry,depthMaterial,null),renderer.renderBufferDirect(shadowCamera,null,geometry,depthMaterial,object,null),object.onAfterShadow(renderer,object,camera,shadowCamera,geometry,depthMaterial,null)}}const children=object.children;for(let i=0,l=children.length;i<l;i++)renderObject(children[i],camera,shadowCamera,light,type)}__name(renderObject,"renderObject");function onMaterialDispose(event){event.target.removeEventListener("dispose",onMaterialDispose);for(const id2 in _materialCache){const cache=_materialCache[id2],uuid=event.target.uuid;uuid in cache&&(cache[uuid].dispose(),delete cache[uuid])}}__name(onMaterialDispose,"onMaterialDispose")}__name(WebGLShadowMap,"WebGLShadowMap");const reversedFuncs={[NeverDepth]:AlwaysDepth,[LessDepth]:GreaterDepth,[EqualDepth]:NotEqualDepth,[LessEqualDepth]:GreaterEqualDepth,[AlwaysDepth]:NeverDepth,[GreaterDepth]:LessDepth,[NotEqualDepth]:EqualDepth,[GreaterEqualDepth]:LessEqualDepth};function WebGLState(gl,extensions){function ColorBuffer(){let locked=!1;const color=new Vector4;let currentColorMask=null;const currentColorClear=new Vector4(0,0,0,0);return{setMask:__name(function(colorMask){currentColorMask!==colorMask&&!locked&&(gl.colorMask(colorMask,colorMask,colorMask,colorMask),currentColorMask=colorMask)},"setMask"),setLocked:__name(function(lock){locked=lock},"setLocked"),setClear:__name(function(r,g,b,a,premultipliedAlpha){premultipliedAlpha===!0&&(r*=a,g*=a,b*=a),color.set(r,g,b,a),currentColorClear.equals(color)===!1&&(gl.clearColor(r,g,b,a),currentColorClear.copy(color))},"setClear"),reset:__name(function(){locked=!1,currentColorMask=null,currentColorClear.set(-1,0,0,0)},"reset")}}__name(ColorBuffer,"ColorBuffer");function DepthBuffer(){let locked=!1,reversed=!1,currentDepthMask=null,currentDepthFunc=null,currentDepthClear=null;return{setReversed:__name(function(value){if(reversed!==value){const ext2=extensions.get("EXT_clip_control");reversed?ext2.clipControlEXT(ext2.LOWER_LEFT_EXT,ext2.ZERO_TO_ONE_EXT):ext2.clipControlEXT(ext2.LOWER_LEFT_EXT,ext2.NEGATIVE_ONE_TO_ONE_EXT);const oldDepth=currentDepthClear;currentDepthClear=null,this.setClear(oldDepth)}reversed=value},"setReversed"),getReversed:__name(function(){return reversed},"getReversed"),setTest:__name(function(depthTest){depthTest?enable(gl.DEPTH_TEST):disable(gl.DEPTH_TEST)},"setTest"),setMask:__name(function(depthMask){currentDepthMask!==depthMask&&!locked&&(gl.depthMask(depthMask),currentDepthMask=depthMask)},"setMask"),setFunc:__name(function(depthFunc){if(reversed&&(depthFunc=reversedFuncs[depthFunc]),currentDepthFunc!==depthFunc){switch(depthFunc){case NeverDepth:gl.depthFunc(gl.NEVER);break;case AlwaysDepth:gl.depthFunc(gl.ALWAYS);break;case LessDepth:gl.depthFunc(gl.LESS);break;case LessEqualDepth:gl.depthFunc(gl.LEQUAL);break;case EqualDepth:gl.depthFunc(gl.EQUAL);break;case GreaterEqualDepth:gl.depthFunc(gl.GEQUAL);break;case GreaterDepth:gl.depthFunc(gl.GREATER);break;case NotEqualDepth:gl.depthFunc(gl.NOTEQUAL);break;default:gl.depthFunc(gl.LEQUAL)}currentDepthFunc=depthFunc}},"setFunc"),setLocked:__name(function(lock){locked=lock},"setLocked"),setClear:__name(function(depth){currentDepthClear!==depth&&(reversed&&(depth=1-depth),gl.clearDepth(depth),currentDepthClear=depth)},"setClear"),reset:__name(function(){locked=!1,currentDepthMask=null,currentDepthFunc=null,currentDepthClear=null,reversed=!1},"reset")}}__name(DepthBuffer,"DepthBuffer");function StencilBuffer(){let locked=!1,currentStencilMask=null,currentStencilFunc=null,currentStencilRef=null,currentStencilFuncMask=null,currentStencilFail=null,currentStencilZFail=null,currentStencilZPass=null,currentStencilClear=null;return{setTest:__name(function(stencilTest){locked||(stencilTest?enable(gl.STENCIL_TEST):disable(gl.STENCIL_TEST))},"setTest"),setMask:__name(function(stencilMask){currentStencilMask!==stencilMask&&!locked&&(gl.stencilMask(stencilMask),currentStencilMask=stencilMask)},"setMask"),setFunc:__name(function(stencilFunc,stencilRef,stencilMask){(currentStencilFunc!==stencilFunc||currentStencilRef!==stencilRef||currentStencilFuncMask!==stencilMask)&&(gl.stencilFunc(stencilFunc,stencilRef,stencilMask),currentStencilFunc=stencilFunc,currentStencilRef=stencilRef,currentStencilFuncMask=stencilMask)},"setFunc"),setOp:__name(function(stencilFail,stencilZFail,stencilZPass){(currentStencilFail!==stencilFail||currentStencilZFail!==stencilZFail||currentStencilZPass!==stencilZPass)&&(gl.stencilOp(stencilFail,stencilZFail,stencilZPass),currentStencilFail=stencilFail,currentStencilZFail=stencilZFail,currentStencilZPass=stencilZPass)},"setOp"),setLocked:__name(function(lock){locked=lock},"setLocked"),setClear:__name(function(stencil){currentStencilClear!==stencil&&(gl.clearStencil(stencil),currentStencilClear=stencil)},"setClear"),reset:__name(function(){locked=!1,currentStencilMask=null,currentStencilFunc=null,currentStencilRef=null,currentStencilFuncMask=null,currentStencilFail=null,currentStencilZFail=null,currentStencilZPass=null,currentStencilClear=null},"reset")}}__name(StencilBuffer,"StencilBuffer");const colorBuffer=new ColorBuffer,depthBuffer=new DepthBuffer,stencilBuffer=new StencilBuffer,uboBindings=new WeakMap,uboProgramMap=new WeakMap;let enabledCapabilities={},currentBoundFramebuffers={},currentDrawbuffers=new WeakMap,defaultDrawbuffers=[],currentProgram=null,currentBlendingEnabled=!1,currentBlending=null,currentBlendEquation=null,currentBlendSrc=null,currentBlendDst=null,currentBlendEquationAlpha=null,currentBlendSrcAlpha=null,currentBlendDstAlpha=null,currentBlendColor=new Color(0,0,0),currentBlendAlpha=0,currentPremultipledAlpha=!1,currentFlipSided=null,currentCullFace=null,currentLineWidth=null,currentPolygonOffsetFactor=null,currentPolygonOffsetUnits=null;const maxTextures=gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let lineWidthAvailable=!1,version=0;const glVersion=gl.getParameter(gl.VERSION);glVersion.indexOf("WebGL")!==-1?(version=parseFloat(/^WebGL (\d)/.exec(glVersion)[1]),lineWidthAvailable=version>=1):glVersion.indexOf("OpenGL ES")!==-1&&(version=parseFloat(/^OpenGL ES (\d)/.exec(glVersion)[1]),lineWidthAvailable=version>=2);let currentTextureSlot=null,currentBoundTextures={};const scissorParam=gl.getParameter(gl.SCISSOR_BOX),viewportParam=gl.getParameter(gl.VIEWPORT),currentScissor=new Vector4().fromArray(scissorParam),currentViewport=new Vector4().fromArray(viewportParam);function createTexture(type,target,count,dimensions){const data=new Uint8Array(4),texture=gl.createTexture();gl.bindTexture(type,texture),gl.texParameteri(type,gl.TEXTURE_MIN_FILTER,gl.NEAREST),gl.texParameteri(type,gl.TEXTURE_MAG_FILTER,gl.NEAREST);for(let i=0;i<count;i++)type===gl.TEXTURE_3D||type===gl.TEXTURE_2D_ARRAY?gl.texImage3D(target,0,gl.RGBA,1,1,dimensions,0,gl.RGBA,gl.UNSIGNED_BYTE,data):gl.texImage2D(target+i,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,data);return texture}__name(createTexture,"createTexture");const emptyTextures={};emptyTextures[gl.TEXTURE_2D]=createTexture(gl.TEXTURE_2D,gl.TEXTURE_2D,1),emptyTextures[gl.TEXTURE_CUBE_MAP]=createTexture(gl.TEXTURE_CUBE_MAP,gl.TEXTURE_CUBE_MAP_POSITIVE_X,6),emptyTextures[gl.TEXTURE_2D_ARRAY]=createTexture(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_2D_ARRAY,1,1),emptyTextures[gl.TEXTURE_3D]=createTexture(gl.TEXTURE_3D,gl.TEXTURE_3D,1,1),colorBuffer.setClear(0,0,0,1),depthBuffer.setClear(1),stencilBuffer.setClear(0),enable(gl.DEPTH_TEST),depthBuffer.setFunc(LessEqualDepth),setFlipSided(!1),setCullFace(CullFaceBack),enable(gl.CULL_FACE),setBlending(NoBlending);function enable(id2){enabledCapabilities[id2]!==!0&&(gl.enable(id2),enabledCapabilities[id2]=!0)}__name(enable,"enable");function disable(id2){enabledCapabilities[id2]!==!1&&(gl.disable(id2),enabledCapabilities[id2]=!1)}__name(disable,"disable");function bindFramebuffer(target,framebuffer){return currentBoundFramebuffers[target]!==framebuffer?(gl.bindFramebuffer(target,framebuffer),currentBoundFramebuffers[target]=framebuffer,target===gl.DRAW_FRAMEBUFFER&&(currentBoundFramebuffers[gl.FRAMEBUFFER]=framebuffer),target===gl.FRAMEBUFFER&&(currentBoundFramebuffers[gl.DRAW_FRAMEBUFFER]=framebuffer),!0):!1}__name(bindFramebuffer,"bindFramebuffer");function drawBuffers(renderTarget,framebuffer){let drawBuffers2=defaultDrawbuffers,needsUpdate=!1;if(renderTarget){drawBuffers2=currentDrawbuffers.get(framebuffer),drawBuffers2===void 0&&(drawBuffers2=[],currentDrawbuffers.set(framebuffer,drawBuffers2));const textures=renderTarget.textures;if(drawBuffers2.length!==textures.length||drawBuffers2[0]!==gl.COLOR_ATTACHMENT0){for(let i=0,il=textures.length;i<il;i++)drawBuffers2[i]=gl.COLOR_ATTACHMENT0+i;drawBuffers2.length=textures.length,needsUpdate=!0}}else drawBuffers2[0]!==gl.BACK&&(drawBuffers2[0]=gl.BACK,needsUpdate=!0);needsUpdate&&gl.drawBuffers(drawBuffers2)}__name(drawBuffers,"drawBuffers");function useProgram(program){return currentProgram!==program?(gl.useProgram(program),currentProgram=program,!0):!1}__name(useProgram,"useProgram");const equationToGL={[AddEquation]:gl.FUNC_ADD,[SubtractEquation]:gl.FUNC_SUBTRACT,[ReverseSubtractEquation]:gl.FUNC_REVERSE_SUBTRACT};equationToGL[MinEquation]=gl.MIN,equationToGL[MaxEquation]=gl.MAX;const factorToGL={[ZeroFactor]:gl.ZERO,[OneFactor]:gl.ONE,[SrcColorFactor]:gl.SRC_COLOR,[SrcAlphaFactor]:gl.SRC_ALPHA,[SrcAlphaSaturateFactor]:gl.SRC_ALPHA_SATURATE,[DstColorFactor]:gl.DST_COLOR,[DstAlphaFactor]:gl.DST_ALPHA,[OneMinusSrcColorFactor]:gl.ONE_MINUS_SRC_COLOR,[OneMinusSrcAlphaFactor]:gl.ONE_MINUS_SRC_ALPHA,[OneMinusDstColorFactor]:gl.ONE_MINUS_DST_COLOR,[OneMinusDstAlphaFactor]:gl.ONE_MINUS_DST_ALPHA,[ConstantColorFactor]:gl.CONSTANT_COLOR,[OneMinusConstantColorFactor]:gl.ONE_MINUS_CONSTANT_COLOR,[ConstantAlphaFactor]:gl.CONSTANT_ALPHA,[OneMinusConstantAlphaFactor]:gl.ONE_MINUS_CONSTANT_ALPHA};function setBlending(blending,blendEquation,blendSrc,blendDst,blendEquationAlpha,blendSrcAlpha,blendDstAlpha,blendColor,blendAlpha,premultipliedAlpha){if(blending===NoBlending){currentBlendingEnabled===!0&&(disable(gl.BLEND),currentBlendingEnabled=!1);return}if(currentBlendingEnabled===!1&&(enable(gl.BLEND),currentBlendingEnabled=!0),blending!==CustomBlending){if(blending!==currentBlending||premultipliedAlpha!==currentPremultipledAlpha){if((currentBlendEquation!==AddEquation||currentBlendEquationAlpha!==AddEquation)&&(gl.blendEquation(gl.FUNC_ADD),currentBlendEquation=AddEquation,currentBlendEquationAlpha=AddEquation),premultipliedAlpha)switch(blending){case NormalBlending:gl.blendFuncSeparate(gl.ONE,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);break;case AdditiveBlending:gl.blendFunc(gl.ONE,gl.ONE);break;case SubtractiveBlending:gl.blendFuncSeparate(gl.ZERO,gl.ONE_MINUS_SRC_COLOR,gl.ZERO,gl.ONE);break;case MultiplyBlending:gl.blendFuncSeparate(gl.ZERO,gl.SRC_COLOR,gl.ZERO,gl.SRC_ALPHA);break;default:console.error("THREE.WebGLState: Invalid blending: ",blending);break}else switch(blending){case NormalBlending:gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);break;case AdditiveBlending:gl.blendFunc(gl.SRC_ALPHA,gl.ONE);break;case SubtractiveBlending:gl.blendFuncSeparate(gl.ZERO,gl.ONE_MINUS_SRC_COLOR,gl.ZERO,gl.ONE);break;case MultiplyBlending:gl.blendFunc(gl.ZERO,gl.SRC_COLOR);break;default:console.error("THREE.WebGLState: Invalid blending: ",blending);break}currentBlendSrc=null,currentBlendDst=null,currentBlendSrcAlpha=null,currentBlendDstAlpha=null,currentBlendColor.set(0,0,0),currentBlendAlpha=0,currentBlending=blending,currentPremultipledAlpha=premultipliedAlpha}return}blendEquationAlpha=blendEquationAlpha||blendEquation,blendSrcAlpha=blendSrcAlpha||blendSrc,blendDstAlpha=blendDstAlpha||blendDst,(blendEquation!==currentBlendEquation||blendEquationAlpha!==currentBlendEquationAlpha)&&(gl.blendEquationSeparate(equationToGL[blendEquation],equationToGL[blendEquationAlpha]),currentBlendEquation=blendEquation,currentBlendEquationAlpha=blendEquationAlpha),(blendSrc!==currentBlendSrc||blendDst!==currentBlendDst||blendSrcAlpha!==currentBlendSrcAlpha||blendDstAlpha!==currentBlendDstAlpha)&&(gl.blendFuncSeparate(factorToGL[blendSrc],factorToGL[blendDst],factorToGL[blendSrcAlpha],factorToGL[blendDstAlpha]),currentBlendSrc=blendSrc,currentBlendDst=blendDst,currentBlendSrcAlpha=blendSrcAlpha,currentBlendDstAlpha=blendDstAlpha),(blendColor.equals(currentBlendColor)===!1||blendAlpha!==currentBlendAlpha)&&(gl.blendColor(blendColor.r,blendColor.g,blendColor.b,blendAlpha),currentBlendColor.copy(blendColor),currentBlendAlpha=blendAlpha),currentBlending=blending,currentPremultipledAlpha=!1}__name(setBlending,"setBlending");function setMaterial(material,frontFaceCW){material.side===DoubleSide?disable(gl.CULL_FACE):enable(gl.CULL_FACE);let flipSided=material.side===BackSide;frontFaceCW&&(flipSided=!flipSided),setFlipSided(flipSided),material.blending===NormalBlending&&material.transparent===!1?setBlending(NoBlending):setBlending(material.blending,material.blendEquation,material.blendSrc,material.blendDst,material.blendEquationAlpha,material.blendSrcAlpha,material.blendDstAlpha,material.blendColor,material.blendAlpha,material.premultipliedAlpha),depthBuffer.setFunc(material.depthFunc),depthBuffer.setTest(material.depthTest),depthBuffer.setMask(material.depthWrite),colorBuffer.setMask(material.colorWrite);const stencilWrite=material.stencilWrite;stencilBuffer.setTest(stencilWrite),stencilWrite&&(stencilBuffer.setMask(material.stencilWriteMask),stencilBuffer.setFunc(material.stencilFunc,material.stencilRef,material.stencilFuncMask),stencilBuffer.setOp(material.stencilFail,material.stencilZFail,material.stencilZPass)),setPolygonOffset(material.polygonOffset,material.polygonOffsetFactor,material.polygonOffsetUnits),material.alphaToCoverage===!0?enable(gl.SAMPLE_ALPHA_TO_COVERAGE):disable(gl.SAMPLE_ALPHA_TO_COVERAGE)}__name(setMaterial,"setMaterial");function setFlipSided(flipSided){currentFlipSided!==flipSided&&(flipSided?gl.frontFace(gl.CW):gl.frontFace(gl.CCW),currentFlipSided=flipSided)}__name(setFlipSided,"setFlipSided");function setCullFace(cullFace){cullFace!==CullFaceNone?(enable(gl.CULL_FACE),cullFace!==currentCullFace&&(cullFace===CullFaceBack?gl.cullFace(gl.BACK):cullFace===CullFaceFront?gl.cullFace(gl.FRONT):gl.cullFace(gl.FRONT_AND_BACK))):disable(gl.CULL_FACE),currentCullFace=cullFace}__name(setCullFace,"setCullFace");function setLineWidth(width){width!==currentLineWidth&&(lineWidthAvailable&&gl.lineWidth(width),currentLineWidth=width)}__name(setLineWidth,"setLineWidth");function setPolygonOffset(polygonOffset,factor,units){polygonOffset?(enable(gl.POLYGON_OFFSET_FILL),(currentPolygonOffsetFactor!==factor||currentPolygonOffsetUnits!==units)&&(gl.polygonOffset(factor,units),currentPolygonOffsetFactor=factor,currentPolygonOffsetUnits=units)):disable(gl.POLYGON_OFFSET_FILL)}__name(setPolygonOffset,"setPolygonOffset");function setScissorTest(scissorTest){scissorTest?enable(gl.SCISSOR_TEST):disable(gl.SCISSOR_TEST)}__name(setScissorTest,"setScissorTest");function activeTexture(webglSlot){webglSlot===void 0&&(webglSlot=gl.TEXTURE0+maxTextures-1),currentTextureSlot!==webglSlot&&(gl.activeTexture(webglSlot),currentTextureSlot=webglSlot)}__name(activeTexture,"activeTexture");function bindTexture(webglType,webglTexture,webglSlot){webglSlot===void 0&&(currentTextureSlot===null?webglSlot=gl.TEXTURE0+maxTextures-1:webglSlot=currentTextureSlot);let boundTexture=currentBoundTextures[webglSlot];boundTexture===void 0&&(boundTexture={type:void 0,texture:void 0},currentBoundTextures[webglSlot]=boundTexture),(boundTexture.type!==webglType||boundTexture.texture!==webglTexture)&&(currentTextureSlot!==webglSlot&&(gl.activeTexture(webglSlot),currentTextureSlot=webglSlot),gl.bindTexture(webglType,webglTexture||emptyTextures[webglType]),boundTexture.type=webglType,boundTexture.texture=webglTexture)}__name(bindTexture,"bindTexture");function unbindTexture(){const boundTexture=currentBoundTextures[currentTextureSlot];boundTexture!==void 0&&boundTexture.type!==void 0&&(gl.bindTexture(boundTexture.type,null),boundTexture.type=void 0,boundTexture.texture=void 0)}__name(unbindTexture,"unbindTexture");function compressedTexImage2D(){try{gl.compressedTexImage2D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(compressedTexImage2D,"compressedTexImage2D");function compressedTexImage3D(){try{gl.compressedTexImage3D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(compressedTexImage3D,"compressedTexImage3D");function texSubImage2D(){try{gl.texSubImage2D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(texSubImage2D,"texSubImage2D");function texSubImage3D(){try{gl.texSubImage3D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(texSubImage3D,"texSubImage3D");function compressedTexSubImage2D(){try{gl.compressedTexSubImage2D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(compressedTexSubImage2D,"compressedTexSubImage2D");function compressedTexSubImage3D(){try{gl.compressedTexSubImage3D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(compressedTexSubImage3D,"compressedTexSubImage3D");function texStorage2D(){try{gl.texStorage2D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(texStorage2D,"texStorage2D");function texStorage3D(){try{gl.texStorage3D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(texStorage3D,"texStorage3D");function texImage2D(){try{gl.texImage2D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(texImage2D,"texImage2D");function texImage3D(){try{gl.texImage3D.apply(gl,arguments)}catch(error){console.error("THREE.WebGLState:",error)}}__name(texImage3D,"texImage3D");function scissor(scissor2){currentScissor.equals(scissor2)===!1&&(gl.scissor(scissor2.x,scissor2.y,scissor2.z,scissor2.w),currentScissor.copy(scissor2))}__name(scissor,"scissor");function viewport(viewport2){currentViewport.equals(viewport2)===!1&&(gl.viewport(viewport2.x,viewport2.y,viewport2.z,viewport2.w),currentViewport.copy(viewport2))}__name(viewport,"viewport");function updateUBOMapping(uniformsGroup,program){let mapping=uboProgramMap.get(program);mapping===void 0&&(mapping=new WeakMap,uboProgramMap.set(program,mapping));let blockIndex=mapping.get(uniformsGroup);blockIndex===void 0&&(blockIndex=gl.getUniformBlockIndex(program,uniformsGroup.name),mapping.set(uniformsGroup,blockIndex))}__name(updateUBOMapping,"updateUBOMapping");function uniformBlockBinding(uniformsGroup,program){const blockIndex=uboProgramMap.get(program).get(uniformsGroup);uboBindings.get(program)!==blockIndex&&(gl.uniformBlockBinding(program,blockIndex,uniformsGroup.__bindingPointIndex),uboBindings.set(program,blockIndex))}__name(uniformBlockBinding,"uniformBlockBinding");function reset(){gl.disable(gl.BLEND),gl.disable(gl.CULL_FACE),gl.disable(gl.DEPTH_TEST),gl.disable(gl.POLYGON_OFFSET_FILL),gl.disable(gl.SCISSOR_TEST),gl.disable(gl.STENCIL_TEST),gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE),gl.blendEquation(gl.FUNC_ADD),gl.blendFunc(gl.ONE,gl.ZERO),gl.blendFuncSeparate(gl.ONE,gl.ZERO,gl.ONE,gl.ZERO),gl.blendColor(0,0,0,0),gl.colorMask(!0,!0,!0,!0),gl.clearColor(0,0,0,0),gl.depthMask(!0),gl.depthFunc(gl.LESS),depthBuffer.setReversed(!1),gl.clearDepth(1),gl.stencilMask(4294967295),gl.stencilFunc(gl.ALWAYS,0,4294967295),gl.stencilOp(gl.KEEP,gl.KEEP,gl.KEEP),gl.clearStencil(0),gl.cullFace(gl.BACK),gl.frontFace(gl.CCW),gl.polygonOffset(0,0),gl.activeTexture(gl.TEXTURE0),gl.bindFramebuffer(gl.FRAMEBUFFER,null),gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null),gl.bindFramebuffer(gl.READ_FRAMEBUFFER,null),gl.useProgram(null),gl.lineWidth(1),gl.scissor(0,0,gl.canvas.width,gl.canvas.height),gl.viewport(0,0,gl.canvas.width,gl.canvas.height),enabledCapabilities={},currentTextureSlot=null,currentBoundTextures={},currentBoundFramebuffers={},currentDrawbuffers=new WeakMap,defaultDrawbuffers=[],currentProgram=null,currentBlendingEnabled=!1,currentBlending=null,currentBlendEquation=null,currentBlendSrc=null,currentBlendDst=null,currentBlendEquationAlpha=null,currentBlendSrcAlpha=null,currentBlendDstAlpha=null,currentBlendColor=new Color(0,0,0),currentBlendAlpha=0,currentPremultipledAlpha=!1,currentFlipSided=null,currentCullFace=null,currentLineWidth=null,currentPolygonOffsetFactor=null,currentPolygonOffsetUnits=null,currentScissor.set(0,0,gl.canvas.width,gl.canvas.height),currentViewport.set(0,0,gl.canvas.width,gl.canvas.height),colorBuffer.reset(),depthBuffer.reset(),stencilBuffer.reset()}return __name(reset,"reset"),{buffers:{color:colorBuffer,depth:depthBuffer,stencil:stencilBuffer},enable,disable,bindFramebuffer,drawBuffers,useProgram,setBlending,setMaterial,setFlipSided,setCullFace,setLineWidth,setPolygonOffset,setScissorTest,activeTexture,bindTexture,unbindTexture,compressedTexImage2D,compressedTexImage3D,texImage2D,texImage3D,updateUBOMapping,uniformBlockBinding,texStorage2D,texStorage3D,texSubImage2D,texSubImage3D,compressedTexSubImage2D,compressedTexSubImage3D,scissor,viewport,reset}}__name(WebGLState,"WebGLState");function getByteLength(width,height,format,type){const typeByteLength=getTextureTypeByteLength(type);switch(format){case AlphaFormat:return width*height;case LuminanceFormat:return width*height;case LuminanceAlphaFormat:return width*height*2;case RedFormat:return width*height/typeByteLength.components*typeByteLength.byteLength;case RedIntegerFormat:return width*height/typeByteLength.components*typeByteLength.byteLength;case RGFormat:return width*height*2/typeByteLength.components*typeByteLength.byteLength;case RGIntegerFormat:return width*height*2/typeByteLength.components*typeByteLength.byteLength;case RGBFormat:return width*height*3/typeByteLength.components*typeByteLength.byteLength;case RGBAFormat:return width*height*4/typeByteLength.components*typeByteLength.byteLength;case RGBAIntegerFormat:return width*height*4/typeByteLength.components*typeByteLength.byteLength;case RGB_S3TC_DXT1_Format:case RGBA_S3TC_DXT1_Format:return Math.floor((width+3)/4)*Math.floor((height+3)/4)*8;case RGBA_S3TC_DXT3_Format:case RGBA_S3TC_DXT5_Format:return Math.floor((width+3)/4)*Math.floor((height+3)/4)*16;case RGB_PVRTC_2BPPV1_Format:case RGBA_PVRTC_2BPPV1_Format:return Math.max(width,16)*Math.max(height,8)/4;case RGB_PVRTC_4BPPV1_Format:case RGBA_PVRTC_4BPPV1_Format:return Math.max(width,8)*Math.max(height,8)/2;case RGB_ETC1_Format:case RGB_ETC2_Format:return Math.floor((width+3)/4)*Math.floor((height+3)/4)*8;case RGBA_ETC2_EAC_Format:return Math.floor((width+3)/4)*Math.floor((height+3)/4)*16;case RGBA_ASTC_4x4_Format:return Math.floor((width+3)/4)*Math.floor((height+3)/4)*16;case RGBA_ASTC_5x4_Format:return Math.floor((width+4)/5)*Math.floor((height+3)/4)*16;case RGBA_ASTC_5x5_Format:return Math.floor((width+4)/5)*Math.floor((height+4)/5)*16;case RGBA_ASTC_6x5_Format:return Math.floor((width+5)/6)*Math.floor((height+4)/5)*16;case RGBA_ASTC_6x6_Format:return Math.floor((width+5)/6)*Math.floor((height+5)/6)*16;case RGBA_ASTC_8x5_Format:return Math.floor((width+7)/8)*Math.floor((height+4)/5)*16;case RGBA_ASTC_8x6_Format:return Math.floor((width+7)/8)*Math.floor((height+5)/6)*16;case RGBA_ASTC_8x8_Format:return Math.floor((width+7)/8)*Math.floor((height+7)/8)*16;case RGBA_ASTC_10x5_Format:return Math.floor((width+9)/10)*Math.floor((height+4)/5)*16;case RGBA_ASTC_10x6_Format:return Math.floor((width+9)/10)*Math.floor((height+5)/6)*16;case RGBA_ASTC_10x8_Format:return Math.floor((width+9)/10)*Math.floor((height+7)/8)*16;case RGBA_ASTC_10x10_Format:return Math.floor((width+9)/10)*Math.floor((height+9)/10)*16;case RGBA_ASTC_12x10_Format:return Math.floor((width+11)/12)*Math.floor((height+9)/10)*16;case RGBA_ASTC_12x12_Format:return Math.floor((width+11)/12)*Math.floor((height+11)/12)*16;case RGBA_BPTC_Format:case RGB_BPTC_SIGNED_Format:case RGB_BPTC_UNSIGNED_Format:return Math.ceil(width/4)*Math.ceil(height/4)*16;case RED_RGTC1_Format:case SIGNED_RED_RGTC1_Format:return Math.ceil(width/4)*Math.ceil(height/4)*8;case RED_GREEN_RGTC2_Format:case SIGNED_RED_GREEN_RGTC2_Format:return Math.ceil(width/4)*Math.ceil(height/4)*16}throw new Error(`Unable to determine texture byte length for ${format} format.`)}__name(getByteLength,"getByteLength");function getTextureTypeByteLength(type){switch(type){case UnsignedByteType:case ByteType:return{byteLength:1,components:1};case UnsignedShortType:case ShortType:case HalfFloatType:return{byteLength:2,components:1};case UnsignedShort4444Type:case UnsignedShort5551Type:return{byteLength:2,components:4};case UnsignedIntType:case IntType:case FloatType:return{byteLength:4,components:1};case UnsignedInt5999Type:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${type}.`)}__name(getTextureTypeByteLength,"getTextureTypeByteLength");function WebGLTextures(_gl,extensions,state,properties,capabilities,utils,info){const multisampledRTTExt=extensions.has("WEBGL_multisampled_render_to_texture")?extensions.get("WEBGL_multisampled_render_to_texture"):null,supportsInvalidateFramebuffer=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),_imageDimensions=new Vector2,_videoTextures=new WeakMap;let _canvas2;const _sources=new WeakMap;let useOffscreenCanvas=!1;try{useOffscreenCanvas=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function createCanvas(width,height){return useOffscreenCanvas?new OffscreenCanvas(width,height):createElementNS("canvas")}__name(createCanvas,"createCanvas");function resizeImage(image,needsNewCanvas,maxSize){let scale=1;const dimensions=getDimensions(image);if((dimensions.width>maxSize||dimensions.height>maxSize)&&(scale=maxSize/Math.max(dimensions.width,dimensions.height)),scale<1)if(typeof HTMLImageElement<"u"&&image instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&image instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&image instanceof ImageBitmap||typeof VideoFrame<"u"&&image instanceof VideoFrame){const width=Math.floor(scale*dimensions.width),height=Math.floor(scale*dimensions.height);_canvas2===void 0&&(_canvas2=createCanvas(width,height));const canvas=needsNewCanvas?createCanvas(width,height):_canvas2;return canvas.width=width,canvas.height=height,canvas.getContext("2d").drawImage(image,0,0,width,height),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+dimensions.width+"x"+dimensions.height+") to ("+width+"x"+height+")."),canvas}else return"data"in image&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+dimensions.width+"x"+dimensions.height+")."),image;return image}__name(resizeImage,"resizeImage");function textureNeedsGenerateMipmaps(texture){return texture.generateMipmaps}__name(textureNeedsGenerateMipmaps,"textureNeedsGenerateMipmaps");function generateMipmap(target){_gl.generateMipmap(target)}__name(generateMipmap,"generateMipmap");function getTargetType(texture){return texture.isWebGLCubeRenderTarget?_gl.TEXTURE_CUBE_MAP:texture.isWebGL3DRenderTarget?_gl.TEXTURE_3D:texture.isWebGLArrayRenderTarget||texture.isCompressedArrayTexture?_gl.TEXTURE_2D_ARRAY:_gl.TEXTURE_2D}__name(getTargetType,"getTargetType");function getInternalFormat(internalFormatName,glFormat,glType,colorSpace,forceLinearTransfer=!1){if(internalFormatName!==null){if(_gl[internalFormatName]!==void 0)return _gl[internalFormatName];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+internalFormatName+"'")}let internalFormat=glFormat;if(glFormat===_gl.RED&&(glType===_gl.FLOAT&&(internalFormat=_gl.R32F),glType===_gl.HALF_FLOAT&&(internalFormat=_gl.R16F),glType===_gl.UNSIGNED_BYTE&&(internalFormat=_gl.R8)),glFormat===_gl.RED_INTEGER&&(glType===_gl.UNSIGNED_BYTE&&(internalFormat=_gl.R8UI),glType===_gl.UNSIGNED_SHORT&&(internalFormat=_gl.R16UI),glType===_gl.UNSIGNED_INT&&(internalFormat=_gl.R32UI),glType===_gl.BYTE&&(internalFormat=_gl.R8I),glType===_gl.SHORT&&(internalFormat=_gl.R16I),glType===_gl.INT&&(internalFormat=_gl.R32I)),glFormat===_gl.RG&&(glType===_gl.FLOAT&&(internalFormat=_gl.RG32F),glType===_gl.HALF_FLOAT&&(internalFormat=_gl.RG16F),glType===_gl.UNSIGNED_BYTE&&(internalFormat=_gl.RG8)),glFormat===_gl.RG_INTEGER&&(glType===_gl.UNSIGNED_BYTE&&(internalFormat=_gl.RG8UI),glType===_gl.UNSIGNED_SHORT&&(internalFormat=_gl.RG16UI),glType===_gl.UNSIGNED_INT&&(internalFormat=_gl.RG32UI),glType===_gl.BYTE&&(internalFormat=_gl.RG8I),glType===_gl.SHORT&&(internalFormat=_gl.RG16I),glType===_gl.INT&&(internalFormat=_gl.RG32I)),glFormat===_gl.RGB_INTEGER&&(glType===_gl.UNSIGNED_BYTE&&(internalFormat=_gl.RGB8UI),glType===_gl.UNSIGNED_SHORT&&(internalFormat=_gl.RGB16UI),glType===_gl.UNSIGNED_INT&&(internalFormat=_gl.RGB32UI),glType===_gl.BYTE&&(internalFormat=_gl.RGB8I),glType===_gl.SHORT&&(internalFormat=_gl.RGB16I),glType===_gl.INT&&(internalFormat=_gl.RGB32I)),glFormat===_gl.RGBA_INTEGER&&(glType===_gl.UNSIGNED_BYTE&&(internalFormat=_gl.RGBA8UI),glType===_gl.UNSIGNED_SHORT&&(internalFormat=_gl.RGBA16UI),glType===_gl.UNSIGNED_INT&&(internalFormat=_gl.RGBA32UI),glType===_gl.BYTE&&(internalFormat=_gl.RGBA8I),glType===_gl.SHORT&&(internalFormat=_gl.RGBA16I),glType===_gl.INT&&(internalFormat=_gl.RGBA32I)),glFormat===_gl.RGB&&glType===_gl.UNSIGNED_INT_5_9_9_9_REV&&(internalFormat=_gl.RGB9_E5),glFormat===_gl.RGBA){const transfer=forceLinearTransfer?LinearTransfer:ColorManagement.getTransfer(colorSpace);glType===_gl.FLOAT&&(internalFormat=_gl.RGBA32F),glType===_gl.HALF_FLOAT&&(internalFormat=_gl.RGBA16F),glType===_gl.UNSIGNED_BYTE&&(internalFormat=transfer===SRGBTransfer?_gl.SRGB8_ALPHA8:_gl.RGBA8),glType===_gl.UNSIGNED_SHORT_4_4_4_4&&(internalFormat=_gl.RGBA4),glType===_gl.UNSIGNED_SHORT_5_5_5_1&&(internalFormat=_gl.RGB5_A1)}return(internalFormat===_gl.R16F||internalFormat===_gl.R32F||internalFormat===_gl.RG16F||internalFormat===_gl.RG32F||internalFormat===_gl.RGBA16F||internalFormat===_gl.RGBA32F)&&extensions.get("EXT_color_buffer_float"),internalFormat}__name(getInternalFormat,"getInternalFormat");function getInternalDepthFormat(useStencil,depthType){let glInternalFormat;return useStencil?depthType===null||depthType===UnsignedIntType||depthType===UnsignedInt248Type?glInternalFormat=_gl.DEPTH24_STENCIL8:depthType===FloatType?glInternalFormat=_gl.DEPTH32F_STENCIL8:depthType===UnsignedShortType&&(glInternalFormat=_gl.DEPTH24_STENCIL8,console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):depthType===null||depthType===UnsignedIntType||depthType===UnsignedInt248Type?glInternalFormat=_gl.DEPTH_COMPONENT24:depthType===FloatType?glInternalFormat=_gl.DEPTH_COMPONENT32F:depthType===UnsignedShortType&&(glInternalFormat=_gl.DEPTH_COMPONENT16),glInternalFormat}__name(getInternalDepthFormat,"getInternalDepthFormat");function getMipLevels(texture,image){return textureNeedsGenerateMipmaps(texture)===!0||texture.isFramebufferTexture&&texture.minFilter!==NearestFilter&&texture.minFilter!==LinearFilter?Math.log2(Math.max(image.width,image.height))+1:texture.mipmaps!==void 0&&texture.mipmaps.length>0?texture.mipmaps.length:texture.isCompressedTexture&&Array.isArray(texture.image)?image.mipmaps.length:1}__name(getMipLevels,"getMipLevels");function onTextureDispose(event){const texture=event.target;texture.removeEventListener("dispose",onTextureDispose),deallocateTexture(texture),texture.isVideoTexture&&_videoTextures.delete(texture)}__name(onTextureDispose,"onTextureDispose");function onRenderTargetDispose(event){const renderTarget=event.target;renderTarget.removeEventListener("dispose",onRenderTargetDispose),deallocateRenderTarget(renderTarget)}__name(onRenderTargetDispose,"onRenderTargetDispose");function deallocateTexture(texture){const textureProperties=properties.get(texture);if(textureProperties.__webglInit===void 0)return;const source=texture.source,webglTextures=_sources.get(source);if(webglTextures){const webglTexture=webglTextures[textureProperties.__cacheKey];webglTexture.usedTimes--,webglTexture.usedTimes===0&&deleteTexture(texture),Object.keys(webglTextures).length===0&&_sources.delete(source)}properties.remove(texture)}__name(deallocateTexture,"deallocateTexture");function deleteTexture(texture){const textureProperties=properties.get(texture);_gl.deleteTexture(textureProperties.__webglTexture);const source=texture.source,webglTextures=_sources.get(source);delete webglTextures[textureProperties.__cacheKey],info.memory.textures--}__name(deleteTexture,"deleteTexture");function deallocateRenderTarget(renderTarget){const renderTargetProperties=properties.get(renderTarget);if(renderTarget.depthTexture&&(renderTarget.depthTexture.dispose(),properties.remove(renderTarget.depthTexture)),renderTarget.isWebGLCubeRenderTarget)for(let i=0;i<6;i++){if(Array.isArray(renderTargetProperties.__webglFramebuffer[i]))for(let level=0;level<renderTargetProperties.__webglFramebuffer[i].length;level++)_gl.deleteFramebuffer(renderTargetProperties.__webglFramebuffer[i][level]);else _gl.deleteFramebuffer(renderTargetProperties.__webglFramebuffer[i]);renderTargetProperties.__webglDepthbuffer&&_gl.deleteRenderbuffer(renderTargetProperties.__webglDepthbuffer[i])}else{if(Array.isArray(renderTargetProperties.__webglFramebuffer))for(let level=0;level<renderTargetProperties.__webglFramebuffer.length;level++)_gl.deleteFramebuffer(renderTargetProperties.__webglFramebuffer[level]);else _gl.deleteFramebuffer(renderTargetProperties.__webglFramebuffer);if(renderTargetProperties.__webglDepthbuffer&&_gl.deleteRenderbuffer(renderTargetProperties.__webglDepthbuffer),renderTargetProperties.__webglMultisampledFramebuffer&&_gl.deleteFramebuffer(renderTargetProperties.__webglMultisampledFramebuffer),renderTargetProperties.__webglColorRenderbuffer)for(let i=0;i<renderTargetProperties.__webglColorRenderbuffer.length;i++)renderTargetProperties.__webglColorRenderbuffer[i]&&_gl.deleteRenderbuffer(renderTargetProperties.__webglColorRenderbuffer[i]);renderTargetProperties.__webglDepthRenderbuffer&&_gl.deleteRenderbuffer(renderTargetProperties.__webglDepthRenderbuffer)}const textures=renderTarget.textures;for(let i=0,il=textures.length;i<il;i++){const attachmentProperties=properties.get(textures[i]);attachmentProperties.__webglTexture&&(_gl.deleteTexture(attachmentProperties.__webglTexture),info.memory.textures--),properties.remove(textures[i])}properties.remove(renderTarget)}__name(deallocateRenderTarget,"deallocateRenderTarget");let textureUnits=0;function resetTextureUnits(){textureUnits=0}__name(resetTextureUnits,"resetTextureUnits");function allocateTextureUnit(){const textureUnit=textureUnits;return textureUnit>=capabilities.maxTextures&&console.warn("THREE.WebGLTextures: Trying to use "+textureUnit+" texture units while this GPU supports only "+capabilities.maxTextures),textureUnits+=1,textureUnit}__name(allocateTextureUnit,"allocateTextureUnit");function getTextureCacheKey(texture){const array=[];return array.push(texture.wrapS),array.push(texture.wrapT),array.push(texture.wrapR||0),array.push(texture.magFilter),array.push(texture.minFilter),array.push(texture.anisotropy),array.push(texture.internalFormat),array.push(texture.format),array.push(texture.type),array.push(texture.generateMipmaps),array.push(texture.premultiplyAlpha),array.push(texture.flipY),array.push(texture.unpackAlignment),array.push(texture.colorSpace),array.join()}__name(getTextureCacheKey,"getTextureCacheKey");function setTexture2D(texture,slot){const textureProperties=properties.get(texture);if(texture.isVideoTexture&&updateVideoTexture(texture),texture.isRenderTargetTexture===!1&&texture.version>0&&textureProperties.__version!==texture.version){const image=texture.image;if(image===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if(image.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{uploadTexture(textureProperties,texture,slot);return}}state.bindTexture(_gl.TEXTURE_2D,textureProperties.__webglTexture,_gl.TEXTURE0+slot)}__name(setTexture2D,"setTexture2D");function setTexture2DArray(texture,slot){const textureProperties=properties.get(texture);if(texture.version>0&&textureProperties.__version!==texture.version){uploadTexture(textureProperties,texture,slot);return}state.bindTexture(_gl.TEXTURE_2D_ARRAY,textureProperties.__webglTexture,_gl.TEXTURE0+slot)}__name(setTexture2DArray,"setTexture2DArray");function setTexture3D(texture,slot){const textureProperties=properties.get(texture);if(texture.version>0&&textureProperties.__version!==texture.version){uploadTexture(textureProperties,texture,slot);return}state.bindTexture(_gl.TEXTURE_3D,textureProperties.__webglTexture,_gl.TEXTURE0+slot)}__name(setTexture3D,"setTexture3D");function setTextureCube(texture,slot){const textureProperties=properties.get(texture);if(texture.version>0&&textureProperties.__version!==texture.version){uploadCubeTexture(textureProperties,texture,slot);return}state.bindTexture(_gl.TEXTURE_CUBE_MAP,textureProperties.__webglTexture,_gl.TEXTURE0+slot)}__name(setTextureCube,"setTextureCube");const wrappingToGL={[RepeatWrapping]:_gl.REPEAT,[ClampToEdgeWrapping]:_gl.CLAMP_TO_EDGE,[MirroredRepeatWrapping]:_gl.MIRRORED_REPEAT},filterToGL={[NearestFilter]:_gl.NEAREST,[NearestMipmapNearestFilter]:_gl.NEAREST_MIPMAP_NEAREST,[NearestMipmapLinearFilter]:_gl.NEAREST_MIPMAP_LINEAR,[LinearFilter]:_gl.LINEAR,[LinearMipmapNearestFilter]:_gl.LINEAR_MIPMAP_NEAREST,[LinearMipmapLinearFilter]:_gl.LINEAR_MIPMAP_LINEAR},compareToGL={[NeverCompare]:_gl.NEVER,[AlwaysCompare]:_gl.ALWAYS,[LessCompare]:_gl.LESS,[LessEqualCompare]:_gl.LEQUAL,[EqualCompare]:_gl.EQUAL,[GreaterEqualCompare]:_gl.GEQUAL,[GreaterCompare]:_gl.GREATER,[NotEqualCompare]:_gl.NOTEQUAL};function setTextureParameters(textureType,texture){if(texture.type===FloatType&&extensions.has("OES_texture_float_linear")===!1&&(texture.magFilter===LinearFilter||texture.magFilter===LinearMipmapNearestFilter||texture.magFilter===NearestMipmapLinearFilter||texture.magFilter===LinearMipmapLinearFilter||texture.minFilter===LinearFilter||texture.minFilter===LinearMipmapNearestFilter||texture.minFilter===NearestMipmapLinearFilter||texture.minFilter===LinearMipmapLinearFilter)&&console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),_gl.texParameteri(textureType,_gl.TEXTURE_WRAP_S,wrappingToGL[texture.wrapS]),_gl.texParameteri(textureType,_gl.TEXTURE_WRAP_T,wrappingToGL[texture.wrapT]),(textureType===_gl.TEXTURE_3D||textureType===_gl.TEXTURE_2D_ARRAY)&&_gl.texParameteri(textureType,_gl.TEXTURE_WRAP_R,wrappingToGL[texture.wrapR]),_gl.texParameteri(textureType,_gl.TEXTURE_MAG_FILTER,filterToGL[texture.magFilter]),_gl.texParameteri(textureType,_gl.TEXTURE_MIN_FILTER,filterToGL[texture.minFilter]),texture.compareFunction&&(_gl.texParameteri(textureType,_gl.TEXTURE_COMPARE_MODE,_gl.COMPARE_REF_TO_TEXTURE),_gl.texParameteri(textureType,_gl.TEXTURE_COMPARE_FUNC,compareToGL[texture.compareFunction])),extensions.has("EXT_texture_filter_anisotropic")===!0){if(texture.magFilter===NearestFilter||texture.minFilter!==NearestMipmapLinearFilter&&texture.minFilter!==LinearMipmapLinearFilter||texture.type===FloatType&&extensions.has("OES_texture_float_linear")===!1)return;if(texture.anisotropy>1||properties.get(texture).__currentAnisotropy){const extension=extensions.get("EXT_texture_filter_anisotropic");_gl.texParameterf(textureType,extension.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(texture.anisotropy,capabilities.getMaxAnisotropy())),properties.get(texture).__currentAnisotropy=texture.anisotropy}}}__name(setTextureParameters,"setTextureParameters");function initTexture(textureProperties,texture){let forceUpload=!1;textureProperties.__webglInit===void 0&&(textureProperties.__webglInit=!0,texture.addEventListener("dispose",onTextureDispose));const source=texture.source;let webglTextures=_sources.get(source);webglTextures===void 0&&(webglTextures={},_sources.set(source,webglTextures));const textureCacheKey=getTextureCacheKey(texture);if(textureCacheKey!==textureProperties.__cacheKey){webglTextures[textureCacheKey]===void 0&&(webglTextures[textureCacheKey]={texture:_gl.createTexture(),usedTimes:0},info.memory.textures++,forceUpload=!0),webglTextures[textureCacheKey].usedTimes++;const webglTexture=webglTextures[textureProperties.__cacheKey];webglTexture!==void 0&&(webglTextures[textureProperties.__cacheKey].usedTimes--,webglTexture.usedTimes===0&&deleteTexture(texture)),textureProperties.__cacheKey=textureCacheKey,textureProperties.__webglTexture=webglTextures[textureCacheKey].texture}return forceUpload}__name(initTexture,"initTexture");function uploadTexture(textureProperties,texture,slot){let textureType=_gl.TEXTURE_2D;(texture.isDataArrayTexture||texture.isCompressedArrayTexture)&&(textureType=_gl.TEXTURE_2D_ARRAY),texture.isData3DTexture&&(textureType=_gl.TEXTURE_3D);const forceUpload=initTexture(textureProperties,texture),source=texture.source;state.bindTexture(textureType,textureProperties.__webglTexture,_gl.TEXTURE0+slot);const sourceProperties=properties.get(source);if(source.version!==sourceProperties.__version||forceUpload===!0){state.activeTexture(_gl.TEXTURE0+slot);const workingPrimaries=ColorManagement.getPrimaries(ColorManagement.workingColorSpace),texturePrimaries=texture.colorSpace===NoColorSpace?null:ColorManagement.getPrimaries(texture.colorSpace),unpackConversion=texture.colorSpace===NoColorSpace||workingPrimaries===texturePrimaries?_gl.NONE:_gl.BROWSER_DEFAULT_WEBGL;_gl.pixelStorei(_gl.UNPACK_FLIP_Y_WEBGL,texture.flipY),_gl.pixelStorei(_gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,texture.premultiplyAlpha),_gl.pixelStorei(_gl.UNPACK_ALIGNMENT,texture.unpackAlignment),_gl.pixelStorei(_gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,unpackConversion);let image=resizeImage(texture.image,!1,capabilities.maxTextureSize);image=verifyColorSpace(texture,image);const glFormat=utils.convert(texture.format,texture.colorSpace),glType=utils.convert(texture.type);let glInternalFormat=getInternalFormat(texture.internalFormat,glFormat,glType,texture.colorSpace,texture.isVideoTexture);setTextureParameters(textureType,texture);let mipmap;const mipmaps=texture.mipmaps,useTexStorage=texture.isVideoTexture!==!0,allocateMemory=sourceProperties.__version===void 0||forceUpload===!0,dataReady=source.dataReady,levels=getMipLevels(texture,image);if(texture.isDepthTexture)glInternalFormat=getInternalDepthFormat(texture.format===DepthStencilFormat,texture.type),allocateMemory&&(useTexStorage?state.texStorage2D(_gl.TEXTURE_2D,1,glInternalFormat,image.width,image.height):state.texImage2D(_gl.TEXTURE_2D,0,glInternalFormat,image.width,image.height,0,glFormat,glType,null));else if(texture.isDataTexture)if(mipmaps.length>0){useTexStorage&&allocateMemory&&state.texStorage2D(_gl.TEXTURE_2D,levels,glInternalFormat,mipmaps[0].width,mipmaps[0].height);for(let i=0,il=mipmaps.length;i<il;i++)mipmap=mipmaps[i],useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_2D,i,0,0,mipmap.width,mipmap.height,glFormat,glType,mipmap.data):state.texImage2D(_gl.TEXTURE_2D,i,glInternalFormat,mipmap.width,mipmap.height,0,glFormat,glType,mipmap.data);texture.generateMipmaps=!1}else useTexStorage?(allocateMemory&&state.texStorage2D(_gl.TEXTURE_2D,levels,glInternalFormat,image.width,image.height),dataReady&&state.texSubImage2D(_gl.TEXTURE_2D,0,0,0,image.width,image.height,glFormat,glType,image.data)):state.texImage2D(_gl.TEXTURE_2D,0,glInternalFormat,image.width,image.height,0,glFormat,glType,image.data);else if(texture.isCompressedTexture)if(texture.isCompressedArrayTexture){useTexStorage&&allocateMemory&&state.texStorage3D(_gl.TEXTURE_2D_ARRAY,levels,glInternalFormat,mipmaps[0].width,mipmaps[0].height,image.depth);for(let i=0,il=mipmaps.length;i<il;i++)if(mipmap=mipmaps[i],texture.format!==RGBAFormat)if(glFormat!==null)if(useTexStorage){if(dataReady)if(texture.layerUpdates.size>0){const layerByteLength=getByteLength(mipmap.width,mipmap.height,texture.format,texture.type);for(const layerIndex of texture.layerUpdates){const layerData=mipmap.data.subarray(layerIndex*layerByteLength/mipmap.data.BYTES_PER_ELEMENT,(layerIndex+1)*layerByteLength/mipmap.data.BYTES_PER_ELEMENT);state.compressedTexSubImage3D(_gl.TEXTURE_2D_ARRAY,i,0,0,layerIndex,mipmap.width,mipmap.height,1,glFormat,layerData)}texture.clearLayerUpdates()}else state.compressedTexSubImage3D(_gl.TEXTURE_2D_ARRAY,i,0,0,0,mipmap.width,mipmap.height,image.depth,glFormat,mipmap.data)}else state.compressedTexImage3D(_gl.TEXTURE_2D_ARRAY,i,glInternalFormat,mipmap.width,mipmap.height,image.depth,0,mipmap.data,0,0);else console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else useTexStorage?dataReady&&state.texSubImage3D(_gl.TEXTURE_2D_ARRAY,i,0,0,0,mipmap.width,mipmap.height,image.depth,glFormat,glType,mipmap.data):state.texImage3D(_gl.TEXTURE_2D_ARRAY,i,glInternalFormat,mipmap.width,mipmap.height,image.depth,0,glFormat,glType,mipmap.data)}else{useTexStorage&&allocateMemory&&state.texStorage2D(_gl.TEXTURE_2D,levels,glInternalFormat,mipmaps[0].width,mipmaps[0].height);for(let i=0,il=mipmaps.length;i<il;i++)mipmap=mipmaps[i],texture.format!==RGBAFormat?glFormat!==null?useTexStorage?dataReady&&state.compressedTexSubImage2D(_gl.TEXTURE_2D,i,0,0,mipmap.width,mipmap.height,glFormat,mipmap.data):state.compressedTexImage2D(_gl.TEXTURE_2D,i,glInternalFormat,mipmap.width,mipmap.height,0,mipmap.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_2D,i,0,0,mipmap.width,mipmap.height,glFormat,glType,mipmap.data):state.texImage2D(_gl.TEXTURE_2D,i,glInternalFormat,mipmap.width,mipmap.height,0,glFormat,glType,mipmap.data)}else if(texture.isDataArrayTexture)if(useTexStorage){if(allocateMemory&&state.texStorage3D(_gl.TEXTURE_2D_ARRAY,levels,glInternalFormat,image.width,image.height,image.depth),dataReady)if(texture.layerUpdates.size>0){const layerByteLength=getByteLength(image.width,image.height,texture.format,texture.type);for(const layerIndex of texture.layerUpdates){const layerData=image.data.subarray(layerIndex*layerByteLength/image.data.BYTES_PER_ELEMENT,(layerIndex+1)*layerByteLength/image.data.BYTES_PER_ELEMENT);state.texSubImage3D(_gl.TEXTURE_2D_ARRAY,0,0,0,layerIndex,image.width,image.height,1,glFormat,glType,layerData)}texture.clearLayerUpdates()}else state.texSubImage3D(_gl.TEXTURE_2D_ARRAY,0,0,0,0,image.width,image.height,image.depth,glFormat,glType,image.data)}else state.texImage3D(_gl.TEXTURE_2D_ARRAY,0,glInternalFormat,image.width,image.height,image.depth,0,glFormat,glType,image.data);else if(texture.isData3DTexture)useTexStorage?(allocateMemory&&state.texStorage3D(_gl.TEXTURE_3D,levels,glInternalFormat,image.width,image.height,image.depth),dataReady&&state.texSubImage3D(_gl.TEXTURE_3D,0,0,0,0,image.width,image.height,image.depth,glFormat,glType,image.data)):state.texImage3D(_gl.TEXTURE_3D,0,glInternalFormat,image.width,image.height,image.depth,0,glFormat,glType,image.data);else if(texture.isFramebufferTexture){if(allocateMemory)if(useTexStorage)state.texStorage2D(_gl.TEXTURE_2D,levels,glInternalFormat,image.width,image.height);else{let width=image.width,height=image.height;for(let i=0;i<levels;i++)state.texImage2D(_gl.TEXTURE_2D,i,glInternalFormat,width,height,0,glFormat,glType,null),width>>=1,height>>=1}}else if(mipmaps.length>0){if(useTexStorage&&allocateMemory){const dimensions=getDimensions(mipmaps[0]);state.texStorage2D(_gl.TEXTURE_2D,levels,glInternalFormat,dimensions.width,dimensions.height)}for(let i=0,il=mipmaps.length;i<il;i++)mipmap=mipmaps[i],useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_2D,i,0,0,glFormat,glType,mipmap):state.texImage2D(_gl.TEXTURE_2D,i,glInternalFormat,glFormat,glType,mipmap);texture.generateMipmaps=!1}else if(useTexStorage){if(allocateMemory){const dimensions=getDimensions(image);state.texStorage2D(_gl.TEXTURE_2D,levels,glInternalFormat,dimensions.width,dimensions.height)}dataReady&&state.texSubImage2D(_gl.TEXTURE_2D,0,0,0,glFormat,glType,image)}else state.texImage2D(_gl.TEXTURE_2D,0,glInternalFormat,glFormat,glType,image);textureNeedsGenerateMipmaps(texture)&&generateMipmap(textureType),sourceProperties.__version=source.version,texture.onUpdate&&texture.onUpdate(texture)}textureProperties.__version=texture.version}__name(uploadTexture,"uploadTexture");function uploadCubeTexture(textureProperties,texture,slot){if(texture.image.length!==6)return;const forceUpload=initTexture(textureProperties,texture),source=texture.source;state.bindTexture(_gl.TEXTURE_CUBE_MAP,textureProperties.__webglTexture,_gl.TEXTURE0+slot);const sourceProperties=properties.get(source);if(source.version!==sourceProperties.__version||forceUpload===!0){state.activeTexture(_gl.TEXTURE0+slot);const workingPrimaries=ColorManagement.getPrimaries(ColorManagement.workingColorSpace),texturePrimaries=texture.colorSpace===NoColorSpace?null:ColorManagement.getPrimaries(texture.colorSpace),unpackConversion=texture.colorSpace===NoColorSpace||workingPrimaries===texturePrimaries?_gl.NONE:_gl.BROWSER_DEFAULT_WEBGL;_gl.pixelStorei(_gl.UNPACK_FLIP_Y_WEBGL,texture.flipY),_gl.pixelStorei(_gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,texture.premultiplyAlpha),_gl.pixelStorei(_gl.UNPACK_ALIGNMENT,texture.unpackAlignment),_gl.pixelStorei(_gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,unpackConversion);const isCompressed=texture.isCompressedTexture||texture.image[0].isCompressedTexture,isDataTexture=texture.image[0]&&texture.image[0].isDataTexture,cubeImage=[];for(let i=0;i<6;i++)!isCompressed&&!isDataTexture?cubeImage[i]=resizeImage(texture.image[i],!0,capabilities.maxCubemapSize):cubeImage[i]=isDataTexture?texture.image[i].image:texture.image[i],cubeImage[i]=verifyColorSpace(texture,cubeImage[i]);const image=cubeImage[0],glFormat=utils.convert(texture.format,texture.colorSpace),glType=utils.convert(texture.type),glInternalFormat=getInternalFormat(texture.internalFormat,glFormat,glType,texture.colorSpace),useTexStorage=texture.isVideoTexture!==!0,allocateMemory=sourceProperties.__version===void 0||forceUpload===!0,dataReady=source.dataReady;let levels=getMipLevels(texture,image);setTextureParameters(_gl.TEXTURE_CUBE_MAP,texture);let mipmaps;if(isCompressed){useTexStorage&&allocateMemory&&state.texStorage2D(_gl.TEXTURE_CUBE_MAP,levels,glInternalFormat,image.width,image.height);for(let i=0;i<6;i++){mipmaps=cubeImage[i].mipmaps;for(let j=0;j<mipmaps.length;j++){const mipmap=mipmaps[j];texture.format!==RGBAFormat?glFormat!==null?useTexStorage?dataReady&&state.compressedTexSubImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j,0,0,mipmap.width,mipmap.height,glFormat,mipmap.data):state.compressedTexImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j,glInternalFormat,mipmap.width,mipmap.height,0,mipmap.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j,0,0,mipmap.width,mipmap.height,glFormat,glType,mipmap.data):state.texImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j,glInternalFormat,mipmap.width,mipmap.height,0,glFormat,glType,mipmap.data)}}}else{if(mipmaps=texture.mipmaps,useTexStorage&&allocateMemory){mipmaps.length>0&&levels++;const dimensions=getDimensions(cubeImage[0]);state.texStorage2D(_gl.TEXTURE_CUBE_MAP,levels,glInternalFormat,dimensions.width,dimensions.height)}for(let i=0;i<6;i++)if(isDataTexture){useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0,0,0,cubeImage[i].width,cubeImage[i].height,glFormat,glType,cubeImage[i].data):state.texImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0,glInternalFormat,cubeImage[i].width,cubeImage[i].height,0,glFormat,glType,cubeImage[i].data);for(let j=0;j<mipmaps.length;j++){const mipmapImage=mipmaps[j].image[i].image;useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j+1,0,0,mipmapImage.width,mipmapImage.height,glFormat,glType,mipmapImage.data):state.texImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j+1,glInternalFormat,mipmapImage.width,mipmapImage.height,0,glFormat,glType,mipmapImage.data)}}else{useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0,0,0,glFormat,glType,cubeImage[i]):state.texImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0,glInternalFormat,glFormat,glType,cubeImage[i]);for(let j=0;j<mipmaps.length;j++){const mipmap=mipmaps[j];useTexStorage?dataReady&&state.texSubImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j+1,0,0,glFormat,glType,mipmap.image[i]):state.texImage2D(_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,j+1,glInternalFormat,glFormat,glType,mipmap.image[i])}}}textureNeedsGenerateMipmaps(texture)&&generateMipmap(_gl.TEXTURE_CUBE_MAP),sourceProperties.__version=source.version,texture.onUpdate&&texture.onUpdate(texture)}textureProperties.__version=texture.version}__name(uploadCubeTexture,"uploadCubeTexture");function setupFrameBufferTexture(framebuffer,renderTarget,texture,attachment,textureTarget,level){const glFormat=utils.convert(texture.format,texture.colorSpace),glType=utils.convert(texture.type),glInternalFormat=getInternalFormat(texture.internalFormat,glFormat,glType,texture.colorSpace),renderTargetProperties=properties.get(renderTarget),textureProperties=properties.get(texture);if(textureProperties.__renderTarget=renderTarget,!renderTargetProperties.__hasExternalTextures){const width=Math.max(1,renderTarget.width>>level),height=Math.max(1,renderTarget.height>>level);textureTarget===_gl.TEXTURE_3D||textureTarget===_gl.TEXTURE_2D_ARRAY?state.texImage3D(textureTarget,level,glInternalFormat,width,height,renderTarget.depth,0,glFormat,glType,null):state.texImage2D(textureTarget,level,glInternalFormat,width,height,0,glFormat,glType,null)}state.bindFramebuffer(_gl.FRAMEBUFFER,framebuffer),useMultisampledRTT(renderTarget)?multisampledRTTExt.framebufferTexture2DMultisampleEXT(_gl.FRAMEBUFFER,attachment,textureTarget,textureProperties.__webglTexture,0,getRenderTargetSamples(renderTarget)):(textureTarget===_gl.TEXTURE_2D||textureTarget>=_gl.TEXTURE_CUBE_MAP_POSITIVE_X&&textureTarget<=_gl.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&_gl.framebufferTexture2D(_gl.FRAMEBUFFER,attachment,textureTarget,textureProperties.__webglTexture,level),state.bindFramebuffer(_gl.FRAMEBUFFER,null)}__name(setupFrameBufferTexture,"setupFrameBufferTexture");function setupRenderBufferStorage(renderbuffer,renderTarget,isMultisample){if(_gl.bindRenderbuffer(_gl.RENDERBUFFER,renderbuffer),renderTarget.depthBuffer){const depthTexture=renderTarget.depthTexture,depthType=depthTexture&&depthTexture.isDepthTexture?depthTexture.type:null,glInternalFormat=getInternalDepthFormat(renderTarget.stencilBuffer,depthType),glAttachmentType=renderTarget.stencilBuffer?_gl.DEPTH_STENCIL_ATTACHMENT:_gl.DEPTH_ATTACHMENT,samples=getRenderTargetSamples(renderTarget);useMultisampledRTT(renderTarget)?multisampledRTTExt.renderbufferStorageMultisampleEXT(_gl.RENDERBUFFER,samples,glInternalFormat,renderTarget.width,renderTarget.height):isMultisample?_gl.renderbufferStorageMultisample(_gl.RENDERBUFFER,samples,glInternalFormat,renderTarget.width,renderTarget.height):_gl.renderbufferStorage(_gl.RENDERBUFFER,glInternalFormat,renderTarget.width,renderTarget.height),_gl.framebufferRenderbuffer(_gl.FRAMEBUFFER,glAttachmentType,_gl.RENDERBUFFER,renderbuffer)}else{const textures=renderTarget.textures;for(let i=0;i<textures.length;i++){const texture=textures[i],glFormat=utils.convert(texture.format,texture.colorSpace),glType=utils.convert(texture.type),glInternalFormat=getInternalFormat(texture.internalFormat,glFormat,glType,texture.colorSpace),samples=getRenderTargetSamples(renderTarget);isMultisample&&useMultisampledRTT(renderTarget)===!1?_gl.renderbufferStorageMultisample(_gl.RENDERBUFFER,samples,glInternalFormat,renderTarget.width,renderTarget.height):useMultisampledRTT(renderTarget)?multisampledRTTExt.renderbufferStorageMultisampleEXT(_gl.RENDERBUFFER,samples,glInternalFormat,renderTarget.width,renderTarget.height):_gl.renderbufferStorage(_gl.RENDERBUFFER,glInternalFormat,renderTarget.width,renderTarget.height)}}_gl.bindRenderbuffer(_gl.RENDERBUFFER,null)}__name(setupRenderBufferStorage,"setupRenderBufferStorage");function setupDepthTexture(framebuffer,renderTarget){if(renderTarget&&renderTarget.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(state.bindFramebuffer(_gl.FRAMEBUFFER,framebuffer),!(renderTarget.depthTexture&&renderTarget.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const textureProperties=properties.get(renderTarget.depthTexture);textureProperties.__renderTarget=renderTarget,(!textureProperties.__webglTexture||renderTarget.depthTexture.image.width!==renderTarget.width||renderTarget.depthTexture.image.height!==renderTarget.height)&&(renderTarget.depthTexture.image.width=renderTarget.width,renderTarget.depthTexture.image.height=renderTarget.height,renderTarget.depthTexture.needsUpdate=!0),setTexture2D(renderTarget.depthTexture,0);const webglDepthTexture=textureProperties.__webglTexture,samples=getRenderTargetSamples(renderTarget);if(renderTarget.depthTexture.format===DepthFormat)useMultisampledRTT(renderTarget)?multisampledRTTExt.framebufferTexture2DMultisampleEXT(_gl.FRAMEBUFFER,_gl.DEPTH_ATTACHMENT,_gl.TEXTURE_2D,webglDepthTexture,0,samples):_gl.framebufferTexture2D(_gl.FRAMEBUFFER,_gl.DEPTH_ATTACHMENT,_gl.TEXTURE_2D,webglDepthTexture,0);else if(renderTarget.depthTexture.format===DepthStencilFormat)useMultisampledRTT(renderTarget)?multisampledRTTExt.framebufferTexture2DMultisampleEXT(_gl.FRAMEBUFFER,_gl.DEPTH_STENCIL_ATTACHMENT,_gl.TEXTURE_2D,webglDepthTexture,0,samples):_gl.framebufferTexture2D(_gl.FRAMEBUFFER,_gl.DEPTH_STENCIL_ATTACHMENT,_gl.TEXTURE_2D,webglDepthTexture,0);else throw new Error("Unknown depthTexture format")}__name(setupDepthTexture,"setupDepthTexture");function setupDepthRenderbuffer(renderTarget){const renderTargetProperties=properties.get(renderTarget),isCube=renderTarget.isWebGLCubeRenderTarget===!0;if(renderTargetProperties.__boundDepthTexture!==renderTarget.depthTexture){const depthTexture=renderTarget.depthTexture;if(renderTargetProperties.__depthDisposeCallback&&renderTargetProperties.__depthDisposeCallback(),depthTexture){const disposeEvent=__name(()=>{delete renderTargetProperties.__boundDepthTexture,delete renderTargetProperties.__depthDisposeCallback,depthTexture.removeEventListener("dispose",disposeEvent)},"disposeEvent");depthTexture.addEventListener("dispose",disposeEvent),renderTargetProperties.__depthDisposeCallback=disposeEvent}renderTargetProperties.__boundDepthTexture=depthTexture}if(renderTarget.depthTexture&&!renderTargetProperties.__autoAllocateDepthBuffer){if(isCube)throw new Error("target.depthTexture not supported in Cube render targets");setupDepthTexture(renderTargetProperties.__webglFramebuffer,renderTarget)}else if(isCube){renderTargetProperties.__webglDepthbuffer=[];for(let i=0;i<6;i++)if(state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglFramebuffer[i]),renderTargetProperties.__webglDepthbuffer[i]===void 0)renderTargetProperties.__webglDepthbuffer[i]=_gl.createRenderbuffer(),setupRenderBufferStorage(renderTargetProperties.__webglDepthbuffer[i],renderTarget,!1);else{const glAttachmentType=renderTarget.stencilBuffer?_gl.DEPTH_STENCIL_ATTACHMENT:_gl.DEPTH_ATTACHMENT,renderbuffer=renderTargetProperties.__webglDepthbuffer[i];_gl.bindRenderbuffer(_gl.RENDERBUFFER,renderbuffer),_gl.framebufferRenderbuffer(_gl.FRAMEBUFFER,glAttachmentType,_gl.RENDERBUFFER,renderbuffer)}}else if(state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglFramebuffer),renderTargetProperties.__webglDepthbuffer===void 0)renderTargetProperties.__webglDepthbuffer=_gl.createRenderbuffer(),setupRenderBufferStorage(renderTargetProperties.__webglDepthbuffer,renderTarget,!1);else{const glAttachmentType=renderTarget.stencilBuffer?_gl.DEPTH_STENCIL_ATTACHMENT:_gl.DEPTH_ATTACHMENT,renderbuffer=renderTargetProperties.__webglDepthbuffer;_gl.bindRenderbuffer(_gl.RENDERBUFFER,renderbuffer),_gl.framebufferRenderbuffer(_gl.FRAMEBUFFER,glAttachmentType,_gl.RENDERBUFFER,renderbuffer)}state.bindFramebuffer(_gl.FRAMEBUFFER,null)}__name(setupDepthRenderbuffer,"setupDepthRenderbuffer");function rebindTextures(renderTarget,colorTexture,depthTexture){const renderTargetProperties=properties.get(renderTarget);colorTexture!==void 0&&setupFrameBufferTexture(renderTargetProperties.__webglFramebuffer,renderTarget,renderTarget.texture,_gl.COLOR_ATTACHMENT0,_gl.TEXTURE_2D,0),depthTexture!==void 0&&setupDepthRenderbuffer(renderTarget)}__name(rebindTextures,"rebindTextures");function setupRenderTarget(renderTarget){const texture=renderTarget.texture,renderTargetProperties=properties.get(renderTarget),textureProperties=properties.get(texture);renderTarget.addEventListener("dispose",onRenderTargetDispose);const textures=renderTarget.textures,isCube=renderTarget.isWebGLCubeRenderTarget===!0,isMultipleRenderTargets=textures.length>1;if(isMultipleRenderTargets||(textureProperties.__webglTexture===void 0&&(textureProperties.__webglTexture=_gl.createTexture()),textureProperties.__version=texture.version,info.memory.textures++),isCube){renderTargetProperties.__webglFramebuffer=[];for(let i=0;i<6;i++)if(texture.mipmaps&&texture.mipmaps.length>0){renderTargetProperties.__webglFramebuffer[i]=[];for(let level=0;level<texture.mipmaps.length;level++)renderTargetProperties.__webglFramebuffer[i][level]=_gl.createFramebuffer()}else renderTargetProperties.__webglFramebuffer[i]=_gl.createFramebuffer()}else{if(texture.mipmaps&&texture.mipmaps.length>0){renderTargetProperties.__webglFramebuffer=[];for(let level=0;level<texture.mipmaps.length;level++)renderTargetProperties.__webglFramebuffer[level]=_gl.createFramebuffer()}else renderTargetProperties.__webglFramebuffer=_gl.createFramebuffer();if(isMultipleRenderTargets)for(let i=0,il=textures.length;i<il;i++){const attachmentProperties=properties.get(textures[i]);attachmentProperties.__webglTexture===void 0&&(attachmentProperties.__webglTexture=_gl.createTexture(),info.memory.textures++)}if(renderTarget.samples>0&&useMultisampledRTT(renderTarget)===!1){renderTargetProperties.__webglMultisampledFramebuffer=_gl.createFramebuffer(),renderTargetProperties.__webglColorRenderbuffer=[],state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglMultisampledFramebuffer);for(let i=0;i<textures.length;i++){const texture2=textures[i];renderTargetProperties.__webglColorRenderbuffer[i]=_gl.createRenderbuffer(),_gl.bindRenderbuffer(_gl.RENDERBUFFER,renderTargetProperties.__webglColorRenderbuffer[i]);const glFormat=utils.convert(texture2.format,texture2.colorSpace),glType=utils.convert(texture2.type),glInternalFormat=getInternalFormat(texture2.internalFormat,glFormat,glType,texture2.colorSpace,renderTarget.isXRRenderTarget===!0),samples=getRenderTargetSamples(renderTarget);_gl.renderbufferStorageMultisample(_gl.RENDERBUFFER,samples,glInternalFormat,renderTarget.width,renderTarget.height),_gl.framebufferRenderbuffer(_gl.FRAMEBUFFER,_gl.COLOR_ATTACHMENT0+i,_gl.RENDERBUFFER,renderTargetProperties.__webglColorRenderbuffer[i])}_gl.bindRenderbuffer(_gl.RENDERBUFFER,null),renderTarget.depthBuffer&&(renderTargetProperties.__webglDepthRenderbuffer=_gl.createRenderbuffer(),setupRenderBufferStorage(renderTargetProperties.__webglDepthRenderbuffer,renderTarget,!0)),state.bindFramebuffer(_gl.FRAMEBUFFER,null)}}if(isCube){state.bindTexture(_gl.TEXTURE_CUBE_MAP,textureProperties.__webglTexture),setTextureParameters(_gl.TEXTURE_CUBE_MAP,texture);for(let i=0;i<6;i++)if(texture.mipmaps&&texture.mipmaps.length>0)for(let level=0;level<texture.mipmaps.length;level++)setupFrameBufferTexture(renderTargetProperties.__webglFramebuffer[i][level],renderTarget,texture,_gl.COLOR_ATTACHMENT0,_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,level);else setupFrameBufferTexture(renderTargetProperties.__webglFramebuffer[i],renderTarget,texture,_gl.COLOR_ATTACHMENT0,_gl.TEXTURE_CUBE_MAP_POSITIVE_X+i,0);textureNeedsGenerateMipmaps(texture)&&generateMipmap(_gl.TEXTURE_CUBE_MAP),state.unbindTexture()}else if(isMultipleRenderTargets){for(let i=0,il=textures.length;i<il;i++){const attachment=textures[i],attachmentProperties=properties.get(attachment);state.bindTexture(_gl.TEXTURE_2D,attachmentProperties.__webglTexture),setTextureParameters(_gl.TEXTURE_2D,attachment),setupFrameBufferTexture(renderTargetProperties.__webglFramebuffer,renderTarget,attachment,_gl.COLOR_ATTACHMENT0+i,_gl.TEXTURE_2D,0),textureNeedsGenerateMipmaps(attachment)&&generateMipmap(_gl.TEXTURE_2D)}state.unbindTexture()}else{let glTextureType=_gl.TEXTURE_2D;if((renderTarget.isWebGL3DRenderTarget||renderTarget.isWebGLArrayRenderTarget)&&(glTextureType=renderTarget.isWebGL3DRenderTarget?_gl.TEXTURE_3D:_gl.TEXTURE_2D_ARRAY),state.bindTexture(glTextureType,textureProperties.__webglTexture),setTextureParameters(glTextureType,texture),texture.mipmaps&&texture.mipmaps.length>0)for(let level=0;level<texture.mipmaps.length;level++)setupFrameBufferTexture(renderTargetProperties.__webglFramebuffer[level],renderTarget,texture,_gl.COLOR_ATTACHMENT0,glTextureType,level);else setupFrameBufferTexture(renderTargetProperties.__webglFramebuffer,renderTarget,texture,_gl.COLOR_ATTACHMENT0,glTextureType,0);textureNeedsGenerateMipmaps(texture)&&generateMipmap(glTextureType),state.unbindTexture()}renderTarget.depthBuffer&&setupDepthRenderbuffer(renderTarget)}__name(setupRenderTarget,"setupRenderTarget");function updateRenderTargetMipmap(renderTarget){const textures=renderTarget.textures;for(let i=0,il=textures.length;i<il;i++){const texture=textures[i];if(textureNeedsGenerateMipmaps(texture)){const targetType=getTargetType(renderTarget),webglTexture=properties.get(texture).__webglTexture;state.bindTexture(targetType,webglTexture),generateMipmap(targetType),state.unbindTexture()}}}__name(updateRenderTargetMipmap,"updateRenderTargetMipmap");const invalidationArrayRead=[],invalidationArrayDraw=[];function updateMultisampleRenderTarget(renderTarget){if(renderTarget.samples>0){if(useMultisampledRTT(renderTarget)===!1){const textures=renderTarget.textures,width=renderTarget.width,height=renderTarget.height;let mask=_gl.COLOR_BUFFER_BIT;const depthStyle=renderTarget.stencilBuffer?_gl.DEPTH_STENCIL_ATTACHMENT:_gl.DEPTH_ATTACHMENT,renderTargetProperties=properties.get(renderTarget),isMultipleRenderTargets=textures.length>1;if(isMultipleRenderTargets)for(let i=0;i<textures.length;i++)state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglMultisampledFramebuffer),_gl.framebufferRenderbuffer(_gl.FRAMEBUFFER,_gl.COLOR_ATTACHMENT0+i,_gl.RENDERBUFFER,null),state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglFramebuffer),_gl.framebufferTexture2D(_gl.DRAW_FRAMEBUFFER,_gl.COLOR_ATTACHMENT0+i,_gl.TEXTURE_2D,null,0);state.bindFramebuffer(_gl.READ_FRAMEBUFFER,renderTargetProperties.__webglMultisampledFramebuffer),state.bindFramebuffer(_gl.DRAW_FRAMEBUFFER,renderTargetProperties.__webglFramebuffer);for(let i=0;i<textures.length;i++){if(renderTarget.resolveDepthBuffer&&(renderTarget.depthBuffer&&(mask|=_gl.DEPTH_BUFFER_BIT),renderTarget.stencilBuffer&&renderTarget.resolveStencilBuffer&&(mask|=_gl.STENCIL_BUFFER_BIT)),isMultipleRenderTargets){_gl.framebufferRenderbuffer(_gl.READ_FRAMEBUFFER,_gl.COLOR_ATTACHMENT0,_gl.RENDERBUFFER,renderTargetProperties.__webglColorRenderbuffer[i]);const webglTexture=properties.get(textures[i]).__webglTexture;_gl.framebufferTexture2D(_gl.DRAW_FRAMEBUFFER,_gl.COLOR_ATTACHMENT0,_gl.TEXTURE_2D,webglTexture,0)}_gl.blitFramebuffer(0,0,width,height,0,0,width,height,mask,_gl.NEAREST),supportsInvalidateFramebuffer===!0&&(invalidationArrayRead.length=0,invalidationArrayDraw.length=0,invalidationArrayRead.push(_gl.COLOR_ATTACHMENT0+i),renderTarget.depthBuffer&&renderTarget.resolveDepthBuffer===!1&&(invalidationArrayRead.push(depthStyle),invalidationArrayDraw.push(depthStyle),_gl.invalidateFramebuffer(_gl.DRAW_FRAMEBUFFER,invalidationArrayDraw)),_gl.invalidateFramebuffer(_gl.READ_FRAMEBUFFER,invalidationArrayRead))}if(state.bindFramebuffer(_gl.READ_FRAMEBUFFER,null),state.bindFramebuffer(_gl.DRAW_FRAMEBUFFER,null),isMultipleRenderTargets)for(let i=0;i<textures.length;i++){state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglMultisampledFramebuffer),_gl.framebufferRenderbuffer(_gl.FRAMEBUFFER,_gl.COLOR_ATTACHMENT0+i,_gl.RENDERBUFFER,renderTargetProperties.__webglColorRenderbuffer[i]);const webglTexture=properties.get(textures[i]).__webglTexture;state.bindFramebuffer(_gl.FRAMEBUFFER,renderTargetProperties.__webglFramebuffer),_gl.framebufferTexture2D(_gl.DRAW_FRAMEBUFFER,_gl.COLOR_ATTACHMENT0+i,_gl.TEXTURE_2D,webglTexture,0)}state.bindFramebuffer(_gl.DRAW_FRAMEBUFFER,renderTargetProperties.__webglMultisampledFramebuffer)}else if(renderTarget.depthBuffer&&renderTarget.resolveDepthBuffer===!1&&supportsInvalidateFramebuffer){const depthStyle=renderTarget.stencilBuffer?_gl.DEPTH_STENCIL_ATTACHMENT:_gl.DEPTH_ATTACHMENT;_gl.invalidateFramebuffer(_gl.DRAW_FRAMEBUFFER,[depthStyle])}}}__name(updateMultisampleRenderTarget,"updateMultisampleRenderTarget");function getRenderTargetSamples(renderTarget){return Math.min(capabilities.maxSamples,renderTarget.samples)}__name(getRenderTargetSamples,"getRenderTargetSamples");function useMultisampledRTT(renderTarget){const renderTargetProperties=properties.get(renderTarget);return renderTarget.samples>0&&extensions.has("WEBGL_multisampled_render_to_texture")===!0&&renderTargetProperties.__useRenderToTexture!==!1}__name(useMultisampledRTT,"useMultisampledRTT");function updateVideoTexture(texture){const frame=info.render.frame;_videoTextures.get(texture)!==frame&&(_videoTextures.set(texture,frame),texture.update())}__name(updateVideoTexture,"updateVideoTexture");function verifyColorSpace(texture,image){const colorSpace=texture.colorSpace,format=texture.format,type=texture.type;return texture.isCompressedTexture===!0||texture.isVideoTexture===!0||colorSpace!==LinearSRGBColorSpace&&colorSpace!==NoColorSpace&&(ColorManagement.getTransfer(colorSpace)===SRGBTransfer?(format!==RGBAFormat||type!==UnsignedByteType)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture color space:",colorSpace)),image}__name(verifyColorSpace,"verifyColorSpace");function getDimensions(image){return typeof HTMLImageElement<"u"&&image instanceof HTMLImageElement?(_imageDimensions.width=image.naturalWidth||image.width,_imageDimensions.height=image.naturalHeight||image.height):typeof VideoFrame<"u"&&image instanceof VideoFrame?(_imageDimensions.width=image.displayWidth,_imageDimensions.height=image.displayHeight):(_imageDimensions.width=image.width,_imageDimensions.height=image.height),_imageDimensions}__name(getDimensions,"getDimensions"),this.allocateTextureUnit=allocateTextureUnit,this.resetTextureUnits=resetTextureUnits,this.setTexture2D=setTexture2D,this.setTexture2DArray=setTexture2DArray,this.setTexture3D=setTexture3D,this.setTextureCube=setTextureCube,this.rebindTextures=rebindTextures,this.setupRenderTarget=setupRenderTarget,this.updateRenderTargetMipmap=updateRenderTargetMipmap,this.updateMultisampleRenderTarget=updateMultisampleRenderTarget,this.setupDepthRenderbuffer=setupDepthRenderbuffer,this.setupFrameBufferTexture=setupFrameBufferTexture,this.useMultisampledRTT=useMultisampledRTT}__name(WebGLTextures,"WebGLTextures");function WebGLUtils(gl,extensions){function convert(p,colorSpace=NoColorSpace){let extension;const transfer=ColorManagement.getTransfer(colorSpace);if(p===UnsignedByteType)return gl.UNSIGNED_BYTE;if(p===UnsignedShort4444Type)return gl.UNSIGNED_SHORT_4_4_4_4;if(p===UnsignedShort5551Type)return gl.UNSIGNED_SHORT_5_5_5_1;if(p===UnsignedInt5999Type)return gl.UNSIGNED_INT_5_9_9_9_REV;if(p===ByteType)return gl.BYTE;if(p===ShortType)return gl.SHORT;if(p===UnsignedShortType)return gl.UNSIGNED_SHORT;if(p===IntType)return gl.INT;if(p===UnsignedIntType)return gl.UNSIGNED_INT;if(p===FloatType)return gl.FLOAT;if(p===HalfFloatType)return gl.HALF_FLOAT;if(p===AlphaFormat)return gl.ALPHA;if(p===RGBFormat)return gl.RGB;if(p===RGBAFormat)return gl.RGBA;if(p===LuminanceFormat)return gl.LUMINANCE;if(p===LuminanceAlphaFormat)return gl.LUMINANCE_ALPHA;if(p===DepthFormat)return gl.DEPTH_COMPONENT;if(p===DepthStencilFormat)return gl.DEPTH_STENCIL;if(p===RedFormat)return gl.RED;if(p===RedIntegerFormat)return gl.RED_INTEGER;if(p===RGFormat)return gl.RG;if(p===RGIntegerFormat)return gl.RG_INTEGER;if(p===RGBAIntegerFormat)return gl.RGBA_INTEGER;if(p===RGB_S3TC_DXT1_Format||p===RGBA_S3TC_DXT1_Format||p===RGBA_S3TC_DXT3_Format||p===RGBA_S3TC_DXT5_Format)if(transfer===SRGBTransfer)if(extension=extensions.get("WEBGL_compressed_texture_s3tc_srgb"),extension!==null){if(p===RGB_S3TC_DXT1_Format)return extension.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(p===RGBA_S3TC_DXT1_Format)return extension.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(p===RGBA_S3TC_DXT3_Format)return extension.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(p===RGBA_S3TC_DXT5_Format)return extension.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(extension=extensions.get("WEBGL_compressed_texture_s3tc"),extension!==null){if(p===RGB_S3TC_DXT1_Format)return extension.COMPRESSED_RGB_S3TC_DXT1_EXT;if(p===RGBA_S3TC_DXT1_Format)return extension.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(p===RGBA_S3TC_DXT3_Format)return extension.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(p===RGBA_S3TC_DXT5_Format)return extension.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(p===RGB_PVRTC_4BPPV1_Format||p===RGB_PVRTC_2BPPV1_Format||p===RGBA_PVRTC_4BPPV1_Format||p===RGBA_PVRTC_2BPPV1_Format)if(extension=extensions.get("WEBGL_compressed_texture_pvrtc"),extension!==null){if(p===RGB_PVRTC_4BPPV1_Format)return extension.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(p===RGB_PVRTC_2BPPV1_Format)return extension.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(p===RGBA_PVRTC_4BPPV1_Format)return extension.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(p===RGBA_PVRTC_2BPPV1_Format)return extension.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(p===RGB_ETC1_Format||p===RGB_ETC2_Format||p===RGBA_ETC2_EAC_Format)if(extension=extensions.get("WEBGL_compressed_texture_etc"),extension!==null){if(p===RGB_ETC1_Format||p===RGB_ETC2_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ETC2:extension.COMPRESSED_RGB8_ETC2;if(p===RGBA_ETC2_EAC_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:extension.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(p===RGBA_ASTC_4x4_Format||p===RGBA_ASTC_5x4_Format||p===RGBA_ASTC_5x5_Format||p===RGBA_ASTC_6x5_Format||p===RGBA_ASTC_6x6_Format||p===RGBA_ASTC_8x5_Format||p===RGBA_ASTC_8x6_Format||p===RGBA_ASTC_8x8_Format||p===RGBA_ASTC_10x5_Format||p===RGBA_ASTC_10x6_Format||p===RGBA_ASTC_10x8_Format||p===RGBA_ASTC_10x10_Format||p===RGBA_ASTC_12x10_Format||p===RGBA_ASTC_12x12_Format)if(extension=extensions.get("WEBGL_compressed_texture_astc"),extension!==null){if(p===RGBA_ASTC_4x4_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:extension.COMPRESSED_RGBA_ASTC_4x4_KHR;if(p===RGBA_ASTC_5x4_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:extension.COMPRESSED_RGBA_ASTC_5x4_KHR;if(p===RGBA_ASTC_5x5_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:extension.COMPRESSED_RGBA_ASTC_5x5_KHR;if(p===RGBA_ASTC_6x5_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:extension.COMPRESSED_RGBA_ASTC_6x5_KHR;if(p===RGBA_ASTC_6x6_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:extension.COMPRESSED_RGBA_ASTC_6x6_KHR;if(p===RGBA_ASTC_8x5_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:extension.COMPRESSED_RGBA_ASTC_8x5_KHR;if(p===RGBA_ASTC_8x6_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:extension.COMPRESSED_RGBA_ASTC_8x6_KHR;if(p===RGBA_ASTC_8x8_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:extension.COMPRESSED_RGBA_ASTC_8x8_KHR;if(p===RGBA_ASTC_10x5_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:extension.COMPRESSED_RGBA_ASTC_10x5_KHR;if(p===RGBA_ASTC_10x6_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:extension.COMPRESSED_RGBA_ASTC_10x6_KHR;if(p===RGBA_ASTC_10x8_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:extension.COMPRESSED_RGBA_ASTC_10x8_KHR;if(p===RGBA_ASTC_10x10_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:extension.COMPRESSED_RGBA_ASTC_10x10_KHR;if(p===RGBA_ASTC_12x10_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:extension.COMPRESSED_RGBA_ASTC_12x10_KHR;if(p===RGBA_ASTC_12x12_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:extension.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(p===RGBA_BPTC_Format||p===RGB_BPTC_SIGNED_Format||p===RGB_BPTC_UNSIGNED_Format)if(extension=extensions.get("EXT_texture_compression_bptc"),extension!==null){if(p===RGBA_BPTC_Format)return transfer===SRGBTransfer?extension.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:extension.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(p===RGB_BPTC_SIGNED_Format)return extension.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(p===RGB_BPTC_UNSIGNED_Format)return extension.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(p===RED_RGTC1_Format||p===SIGNED_RED_RGTC1_Format||p===RED_GREEN_RGTC2_Format||p===SIGNED_RED_GREEN_RGTC2_Format)if(extension=extensions.get("EXT_texture_compression_rgtc"),extension!==null){if(p===RGBA_BPTC_Format)return extension.COMPRESSED_RED_RGTC1_EXT;if(p===SIGNED_RED_RGTC1_Format)return extension.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(p===RED_GREEN_RGTC2_Format)return extension.COMPRESSED_RED_GREEN_RGTC2_EXT;if(p===SIGNED_RED_GREEN_RGTC2_Format)return extension.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return p===UnsignedInt248Type?gl.UNSIGNED_INT_24_8:gl[p]!==void 0?gl[p]:null}return __name(convert,"convert"),{convert}}__name(WebGLUtils,"WebGLUtils");class ArrayCamera extends PerspectiveCamera{static{__name(this,"ArrayCamera")}constructor(array=[]){super(),this.isArrayCamera=!0,this.cameras=array}}class Group extends Object3D{static{__name(this,"Group")}constructor(){super(),this.isGroup=!0,this.type="Group"}}const _moveEvent={type:"move"};class WebXRController{static{__name(this,"WebXRController")}constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Group,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Group,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new Vector3,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new Vector3),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Group,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new Vector3,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new Vector3),this._grip}dispatchEvent(event){return this._targetRay!==null&&this._targetRay.dispatchEvent(event),this._grip!==null&&this._grip.dispatchEvent(event),this._hand!==null&&this._hand.dispatchEvent(event),this}connect(inputSource){if(inputSource&&inputSource.hand){const hand=this._hand;if(hand)for(const inputjoint of inputSource.hand.values())this._getHandJoint(hand,inputjoint)}return this.dispatchEvent({type:"connected",data:inputSource}),this}disconnect(inputSource){return this.dispatchEvent({type:"disconnected",data:inputSource}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(inputSource,frame,referenceSpace){let inputPose=null,gripPose=null,handPose=null;const targetRay=this._targetRay,grip=this._grip,hand=this._hand;if(inputSource&&frame.session.visibilityState!=="visible-blurred"){if(hand&&inputSource.hand){handPose=!0;for(const inputjoint of inputSource.hand.values()){const jointPose=frame.getJointPose(inputjoint,referenceSpace),joint=this._getHandJoint(hand,inputjoint);jointPose!==null&&(joint.matrix.fromArray(jointPose.transform.matrix),joint.matrix.decompose(joint.position,joint.rotation,joint.scale),joint.matrixWorldNeedsUpdate=!0,joint.jointRadius=jointPose.radius),joint.visible=jointPose!==null}const indexTip=hand.joints["index-finger-tip"],thumbTip=hand.joints["thumb-tip"],distance=indexTip.position.distanceTo(thumbTip.position),distanceToPinch=.02,threshold=.005;hand.inputState.pinching&&distance>distanceToPinch+threshold?(hand.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:inputSource.handedness,target:this})):!hand.inputState.pinching&&distance<=distanceToPinch-threshold&&(hand.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:inputSource.handedness,target:this}))}else grip!==null&&inputSource.gripSpace&&(gripPose=frame.getPose(inputSource.gripSpace,referenceSpace),gripPose!==null&&(grip.matrix.fromArray(gripPose.transform.matrix),grip.matrix.decompose(grip.position,grip.rotation,grip.scale),grip.matrixWorldNeedsUpdate=!0,gripPose.linearVelocity?(grip.hasLinearVelocity=!0,grip.linearVelocity.copy(gripPose.linearVelocity)):grip.hasLinearVelocity=!1,gripPose.angularVelocity?(grip.hasAngularVelocity=!0,grip.angularVelocity.copy(gripPose.angularVelocity)):grip.hasAngularVelocity=!1));targetRay!==null&&(inputPose=frame.getPose(inputSource.targetRaySpace,referenceSpace),inputPose===null&&gripPose!==null&&(inputPose=gripPose),inputPose!==null&&(targetRay.matrix.fromArray(inputPose.transform.matrix),targetRay.matrix.decompose(targetRay.position,targetRay.rotation,targetRay.scale),targetRay.matrixWorldNeedsUpdate=!0,inputPose.linearVelocity?(targetRay.hasLinearVelocity=!0,targetRay.linearVelocity.copy(inputPose.linearVelocity)):targetRay.hasLinearVelocity=!1,inputPose.angularVelocity?(targetRay.hasAngularVelocity=!0,targetRay.angularVelocity.copy(inputPose.angularVelocity)):targetRay.hasAngularVelocity=!1,this.dispatchEvent(_moveEvent)))}return targetRay!==null&&(targetRay.visible=inputPose!==null),grip!==null&&(grip.visible=gripPose!==null),hand!==null&&(hand.visible=handPose!==null),this}_getHandJoint(hand,inputjoint){if(hand.joints[inputjoint.jointName]===void 0){const joint=new Group;joint.matrixAutoUpdate=!1,joint.visible=!1,hand.joints[inputjoint.jointName]=joint,hand.add(joint)}return hand.joints[inputjoint.jointName]}}const _occlusion_vertex=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,_occlusion_fragment=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

`)!==-1&&(text=text.replace(/\r\n/g,`
`)),text.indexOf(`\\
`)!==-1&&(text=text.replace(/\\\n/g,""));const lines=text.split(`
`);let result=[];for(let i=0,l=lines.length;i<l;i++){const line=lines[i].trimStart();if(line.length===0)continue;const lineFirstChar=line.charAt(0);if(lineFirstChar!=="#")if(lineFirstChar==="v"){const data=line.split(_face_vertex_data_separator_pattern);switch(data[0]){case"v":state.vertices.push(parseFloat(data[1]),parseFloat(data[2]),parseFloat(data[3])),data.length>=7?(_color.setRGB(parseFloat(data[4]),parseFloat(data[5]),parseFloat(data[6]),SRGBColorSpace),state.colors.push(_color.r,_color.g,_color.b)):state.colors.push(void 0,void 0,void 0);break;case"vn":state.normals.push(parseFloat(data[1]),parseFloat(data[2]),parseFloat(data[3]));break;case"vt":state.uvs.push(parseFloat(data[1]),parseFloat(data[2]));break}}else if(lineFirstChar==="f"){const vertexData=line.slice(1).trim().split(_face_vertex_data_separator_pattern),faceVertices=[];for(let j=0,jl=vertexData.length;j<jl;j++){const vertex2=vertexData[j];if(vertex2.length>0){const vertexParts=vertex2.split("/");faceVertices.push(vertexParts)}}const v1=faceVertices[0];for(let j=1,jl=faceVertices.length-1;j<jl;j++){const v2=faceVertices[j],v3=faceVertices[j+1];state.addFace(v1[0],v2[0],v3[0],v1[1],v2[1],v3[1],v1[2],v2[2],v3[2])}}else if(lineFirstChar==="l"){const lineParts=line.substring(1).trim().split(" ");let lineVertices=[];const lineUVs=[];if(line.indexOf("/")===-1)lineVertices=lineParts;else for(let li=0,llen=lineParts.length;li<llen;li++){const parts=lineParts[li].split("/");parts[0]!==""&&lineVertices.push(parts[0]),parts[1]!==""&&lineUVs.push(parts[1])}state.addLineGeometry(lineVertices,lineUVs)}else if(lineFirstChar==="p"){const pointData=line.slice(1).trim().split(" ");state.addPointGeometry(pointData)}else if((result=_object_pattern.exec(line))!==null){const name=(" "+result[0].slice(1).trim()).slice(1);state.startObject(name)}else if(_material_use_pattern.test(line))state.object.startMaterial(line.substring(7).trim(),state.materialLibraries);else if(_material_library_pattern.test(line))state.materialLibraries.push(line.substring(7).trim());else if(_map_use_pattern.test(line))console.warn('THREE.OBJLoader: Rendering identifier "usemap" not supported. Textures must be defined in MTL files.');else if(lineFirstChar==="s"){if(result=line.split(" "),result.length>1){const value=result[1].trim().toLowerCase();state.object.smooth=value!=="0"&&value!=="off"}else state.object.smooth=!0;const material=state.object.currentMaterial();material&&(material.smooth=state.object.smooth)}else{if(line==="\0")continue;console.warn('THREE.OBJLoader: Unexpected line: "'+line+'"')}}state.finalize();const container=new Group;if(container.materialLibraries=[].concat(state.materialLibraries),!(state.objects.length===1&&state.objects[0].geometry.vertices.length===0)===!0)for(let i=0,l=state.objects.length;i<l;i++){const object=state.objects[i],geometry=object.geometry,materials=object.materials,isLine=geometry.type==="Line",isPoints=geometry.type==="Points";let hasVertexColors=!1;if(geometry.vertices.length===0)continue;const buffergeometry=new BufferGeometry;buffergeometry.setAttribute("position",new Float32BufferAttribute(geometry.vertices,3)),geometry.normals.length>0&&buffergeometry.setAttribute("normal",new Float32BufferAttribute(geometry.normals,3)),geometry.colors.length>0&&(hasVertexColors=!0,buffergeometry.setAttribute("color",new Float32BufferAttribute(geometry.colors,3))),geometry.hasUVIndices===!0&&buffergeometry.setAttribute("uv",new Float32BufferAttribute(geometry.uvs,2));const createdMaterials=[];for(let mi=0,miLen=materials.length;mi<miLen;mi++){const sourceMaterial=materials[mi],materialHash=sourceMaterial.name+"_"+sourceMaterial.smooth+"_"+hasVertexColors;let material=state.materials[materialHash];if(this.materials!==null){if(material=this.materials.create(sourceMaterial.name),isLine&&material&&!(material instanceof LineBasicMaterial)){const materialLine=new LineBasicMaterial;Material.prototype.copy.call(materialLine,material),materialLine.color.copy(material.color),material=materialLine}else if(isPoints&&material&&!(material instanceof PointsMaterial)){const materialPoints=new PointsMaterial({size:10,sizeAttenuation:!1});Material.prototype.copy.call(materialPoints,material),materialPoints.color.copy(material.color),materialPoints.map=material.map,material=materialPoints}}material===void 0&&(isLine?material=new LineBasicMaterial:isPoints?material=new PointsMaterial({size:1,sizeAttenuation:!1}):material=new MeshPhongMaterial,material.name=sourceMaterial.name,material.flatShading=!sourceMaterial.smooth,material.vertexColors=hasVertexColors,state.materials[materialHash]=material),createdMaterials.push(material)}let mesh;if(createdMaterials.length>1){for(let mi=0,miLen=materials.length;mi<miLen;mi++){const sourceMaterial=materials[mi];buffergeometry.addGroup(sourceMaterial.groupStart,sourceMaterial.groupCount,mi)}isLine?mesh=new LineSegments(buffergeometry,createdMaterials):isPoints?mesh=new Points(buffergeometry,createdMaterials):mesh=new Mesh(buffergeometry,createdMaterials)}else isLine?mesh=new LineSegments(buffergeometry,createdMaterials[0]):isPoints?mesh=new Points(buffergeometry,createdMaterials[0]):mesh=new Mesh(buffergeometry,createdMaterials[0]);mesh.name=object.name,container.add(mesh)}else if(state.vertices.length>0){const material=new PointsMaterial({size:1,sizeAttenuation:!1}),buffergeometry=new BufferGeometry;buffergeometry.setAttribute("position",new Float32BufferAttribute(state.vertices,3)),state.colors.length>0&&state.colors[0]!==void 0&&(buffergeometry.setAttribute("color",new Float32BufferAttribute(state.colors,3)),material.vertexColors=!0);const points=new Points(buffergeometry,material);container.add(points)}return container}}class MTLLoader extends Loader{static{__name(this,"MTLLoader")}constructor(manager){super(manager)}load(url,onLoad,onProgress,onError){const scope=this,path=this.path===""?LoaderUtils.extractUrlBase(url):this.path,loader=new FileLoader(this.manager);loader.setPath(this.path),loader.setRequestHeader(this.requestHeader),loader.setWithCredentials(this.withCredentials),loader.load(url,function(text){try{onLoad(scope.parse(text,path))}catch(e){onError?onError(e):console.error(e),scope.manager.itemError(url)}},onProgress,onError)}setMaterialOptions(value){return this.materialOptions=value,this}parse(text,path){const lines=text.split(`
`);let info={};const delimiter_pattern=/\s+/,materialsInfo={};for(let i=0;i<lines.length;i++){let line=lines[i];if(line=line.trim(),line.length===0||line.charAt(0)==="#")continue;const pos=line.indexOf(" ");let key=pos>=0?line.substring(0,pos):line;key=key.toLowerCase();let value=pos>=0?line.substring(pos+1):"";if(value=value.trim(),key==="newmtl")info={name:value},materialsInfo[value]=info;else if(key==="ka"||key==="kd"||key==="ks"||key==="ke"){const ss=value.split(delimiter_pattern,3);info[key]=[parseFloat(ss[0]),parseFloat(ss[1]),parseFloat(ss[2])]}else info[key]=value}const materialCreator=new MaterialCreator(this.resourcePath||path,this.materialOptions);return materialCreator.setCrossOrigin(this.crossOrigin),materialCreator.setManager(this.manager),materialCreator.setMaterials(materialsInfo),materialCreator}}class MaterialCreator{static{__name(this,"MaterialCreator")}constructor(baseUrl="",options={}){this.baseUrl=baseUrl,this.options=options,this.materialsInfo={},this.materials={},this.materialsArray=[],this.nameLookup={},this.crossOrigin="anonymous",this.side=this.options.side!==void 0?this.options.side:FrontSide,this.wrap=this.options.wrap!==void 0?this.options.wrap:RepeatWrapping}setCrossOrigin(value){return this.crossOrigin=value,this}setManager(value){this.manager=value}setMaterials(materialsInfo){this.materialsInfo=this.convert(materialsInfo),this.materials={},this.materialsArray=[],this.nameLookup={}}convert(materialsInfo){if(!this.options)return materialsInfo;const converted={};for(const mn in materialsInfo){const mat=materialsInfo[mn],covmat={};converted[mn]=covmat;for(const prop in mat){let save=!0,value=mat[prop];const lprop=prop.toLowerCase();switch(lprop){case"kd":case"ka":case"ks":this.options&&this.options.normalizeRGB&&(value=[value[0]/255,value[1]/255,value[2]/255]),this.options&&this.options.ignoreZeroRGBs&&value[0]===0&&value[1]===0&&value[2]===0&&(save=!1);break;default:break}save&&(covmat[lprop]=value)}}return converted}preload(){for(const mn in this.materialsInfo)this.create(mn)}getIndex(materialName){return this.nameLookup[materialName]}getAsArray(){let index=0;for(const mn in this.materialsInfo)this.materialsArray[index]=this.create(mn),this.nameLookup[mn]=index,index++;return this.materialsArray}create(materialName){return this.materials[materialName]===void 0&&this.createMaterial_(materialName),this.materials[materialName]}createMaterial_(materialName){const scope=this,mat=this.materialsInfo[materialName],params={name:materialName,side:this.side};function resolveURL(baseUrl,url){return typeof url!="string"||url===""?"":/^https?:\/\//i.test(url)?url:baseUrl+url}__name(resolveURL,"resolveURL");function setMapForType(mapType,value){if(params[mapType])return;const texParams=scope.getTextureParams(value,params),map=scope.loadTexture(resolveURL(scope.baseUrl,texParams.url));map.repeat.copy(texParams.scale),map.offset.copy(texParams.offset),map.wrapS=scope.wrap,map.wrapT=scope.wrap,(mapType==="map"||mapType==="emissiveMap")&&(map.colorSpace=SRGBColorSpace),params[mapType]=map}__name(setMapForType,"setMapForType");for(const prop in mat){const value=mat[prop];let n;if(value!=="")switch(prop.toLowerCase()){case"kd":params.color=ColorManagement.toWorkingColorSpace(new Color().fromArray(value),SRGBColorSpace);break;case"ks":params.specular=ColorManagement.toWorkingColorSpace(new Color().fromArray(value),SRGBColorSpace);break;case"ke":params.emissive=ColorManagement.toWorkingColorSpace(new Color().fromArray(value),SRGBColorSpace);break;case"map_kd":setMapForType("map",value);break;case"map_ks":setMapForType("specularMap",value);break;case"map_ke":setMapForType("emissiveMap",value);break;case"norm":setMapForType("normalMap",value);break;case"map_bump":case"bump":setMapForType("bumpMap",value);break;case"map_d":setMapForType("alphaMap",value),params.transparent=!0;break;case"ns":params.shininess=parseFloat(value);break;case"d":n=parseFloat(value),n<1&&(params.opacity=n,params.transparent=!0);break;case"tr":n=parseFloat(value),this.options&&this.options.invertTrProperty&&(n=1-n),n>0&&(params.opacity=1-n,params.transparent=!0);break;default:break}}return this.materials[materialName]=new MeshPhongMaterial(params),this.materials[materialName]}getTextureParams(value,matParams){const texParams={scale:new Vector2(1,1),offset:new Vector2(0,0)},items=value.split(/\s+/);let pos;return pos=items.indexOf("-bm"),pos>=0&&(matParams.bumpScale=parseFloat(items[pos+1]),items.splice(pos,2)),pos=items.indexOf("-s"),pos>=0&&(texParams.scale.set(parseFloat(items[pos+1]),parseFloat(items[pos+2])),items.splice(pos,4)),pos=items.indexOf("-o"),pos>=0&&(texParams.offset.set(parseFloat(items[pos+1]),parseFloat(items[pos+2])),items.splice(pos,4)),texParams.url=items.join(" ").trim(),texParams}loadTexture(url,mapping,onLoad,onProgress,onError){const manager=this.manager!==void 0?this.manager:DefaultLoadingManager;let loader=manager.getHandler(url);loader===null&&(loader=new TextureLoader(manager)),loader.setCrossOrigin&&loader.setCrossOrigin(this.crossOrigin);const texture=loader.load(url,onLoad,onProgress,onError);return mapping!==void 0&&(texture.mapping=mapping),texture}}/*!
fflate - fast JavaScript compression/decompression
<https://101arrowz.github.io/fflate>
Licensed under MIT. https://github.com/101arrowz/fflate/blob/master/LICENSE
version 0.8.2
*/var u8=Uint8Array,u16=Uint16Array,i32=Int32Array,fleb=new u8([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),fdeb=new u8([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),clim=new u8([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),freb=__name(function(eb,start){for(var b=new u16(31),i=0;i<31;++i)b[i]=start+=1<<eb[i-1];for(var r=new i32(b[30]),i=1;i<30;++i)for(var j=b[i];j<b[i+1];++j)r[j]=j-b[i]<<5|i;return{b,r}},"freb"),_a=freb(fleb,2),fl=_a.b,revfl=_a.r;fl[28]=258,revfl[258]=28;var _b=freb(fdeb,0),fd=_b.b,revfd=_b.r,rev=new u16(32768);for(var i=0;i<32768;++i){var x=(i&43690)>>1|(i&21845)<<1;x=(x&52428)>>2|(x&13107)<<2,x=(x&61680)>>4|(x&3855)<<4,rev[i]=((x&65280)>>8|(x&255)<<8)>>1}var hMap=__name(function(cd,mb,r){for(var s=cd.length,i=0,l=new u16(mb);i<s;++i)cd[i]&&++l[cd[i]-1];var le=new u16(mb);for(i=1;i<mb;++i)le[i]=le[i-1]+l[i-1]<<1;var co;if(r){co=new u16(1<<mb);var rvb=15-mb;for(i=0;i<s;++i)if(cd[i])for(var sv=i<<4|cd[i],r_1=mb-cd[i],v=le[cd[i]-1]++<<r_1,m=v|(1<<r_1)-1;v<=m;++v)co[rev[v]>>rvb]=sv}else for(co=new u16(s),i=0;i<s;++i)cd[i]&&(co[i]=rev[le[cd[i]-1]++]>>15-cd[i]);return co},"hMap"),flt=new u8(288);for(var i=0;i<144;++i)flt[i]=8;for(var i=144;i<256;++i)flt[i]=9;for(var i=256;i<280;++i)flt[i]=7;for(var i=280;i<288;++i)flt[i]=8;var fdt=new u8(32);for(var i=0;i<32;++i)fdt[i]=5;var flrm=hMap(flt,9,1);var fdrm=hMap(fdt,5,1),max=__name(function(a){for(var m=a[0],i=1;i<a.length;++i)a[i]>m&&(m=a[i]);return m},"max"),bits=__name(function(d,p,m){var o=p/8|0;return(d[o]|d[o+1]<<8)>>(p&7)&m},"bits"),bits16=__name(function(d,p){var o=p/8|0;return(d[o]|d[o+1]<<8|d[o+2]<<16)>>(p&7)},"bits16"),shft=__name(function(p){return(p+7)/8|0},"shft"),slc=__name(function(v,s,e){return(s==null||s<0)&&(s=0),(e==null||e>v.length)&&(e=v.length),new u8(v.subarray(s,e))},"slc");var ec=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long","stream finishing","invalid zip data"],err=__name(function(ind,msg,nt){var e=new Error(msg||ec[ind]);if(e.code=ind,Error.captureStackTrace&&Error.captureStackTrace(e,err),!nt)throw e;return e},"err"),inflt=__name(function(dat,st,buf,dict){var sl=dat.length,dl=dict?dict.length:0;if(!sl||st.f&&!st.l)return buf||new u8(0);var noBuf=!buf,resize=noBuf||st.i!=2,noSt=st.i;noBuf&&(buf=new u8(sl*3));var cbuf=__name(function(l2){var bl=buf.length;if(l2>bl){var nbuf=new u8(Math.max(bl*2,l2));nbuf.set(buf),buf=nbuf}},"cbuf"),final=st.f||0,pos=st.p||0,bt=st.b||0,lm=st.l,dm=st.d,lbt=st.m,dbt=st.n,tbts=sl*8;do{if(!lm){final=bits(dat,pos,1);var type=bits(dat,pos+1,3);if(pos+=3,type)if(type==1)lm=flrm,dm=fdrm,lbt=9,dbt=5;else if(type==2){var hLit=bits(dat,pos,31)+257,hcLen=bits(dat,pos+10,15)+4,tl=hLit+bits(dat,pos+5,31)+1;pos+=14;for(var ldt=new u8(tl),clt=new u8(19),i=0;i<hcLen;++i)clt[clim[i]]=bits(dat,pos+i*3,7);pos+=hcLen*3;for(var clb=max(clt),clbmsk=(1<<clb)-1,clm=hMap(clt,clb,1),i=0;i<tl;){var r=clm[bits(dat,pos,clbmsk)];pos+=r&15;var s=r>>4;if(s<16)ldt[i++]=s;else{var c=0,n=0;for(s==16?(n=3+bits(dat,pos,3),pos+=2,c=ldt[i-1]):s==17?(n=3+bits(dat,pos,7),pos+=3):s==18&&(n=11+bits(dat,pos,127),pos+=7);n--;)ldt[i++]=c}}var lt=ldt.subarray(0,hLit),dt=ldt.subarray(hLit);lbt=max(lt),dbt=max(dt),lm=hMap(lt,lbt,1),dm=hMap(dt,dbt,1)}else err(1);else{var s=shft(pos)+4,l=dat[s-4]|dat[s-3]<<8,t2=s+l;if(t2>sl){noSt&&err(0);break}resize&&cbuf(bt+l),buf.set(dat.subarray(s,t2),bt),st.b=bt+=l,st.p=pos=t2*8,st.f=final;continue}if(pos>tbts){noSt&&err(0);break}}resize&&cbuf(bt+131072);for(var lms=(1<<lbt)-1,dms=(1<<dbt)-1,lpos=pos;;lpos=pos){var c=lm[bits16(dat,pos)&lms],sym=c>>4;if(pos+=c&15,pos>tbts){noSt&&err(0);break}if(c||err(2),sym<256)buf[bt++]=sym;else if(sym==256){lpos=pos,lm=null;break}else{var add=sym-254;if(sym>264){var i=sym-257,b=fleb[i];add=bits(dat,pos,(1<<b)-1)+fl[i],pos+=b}var d=dm[bits16(dat,pos)&dms],dsym=d>>4;d||err(3),pos+=d&15;var dt=fd[dsym];if(dsym>3){var b=fdeb[dsym];dt+=bits16(dat,pos)&(1<<b)-1,pos+=b}if(pos>tbts){noSt&&err(0);break}resize&&cbuf(bt+131072);var end=bt+add;if(bt<dt){var shift=dl-dt,dend=Math.min(dt,end);for(shift+bt<0&&err(3);bt<dend;++bt)buf[bt]=dict[shift+bt]}for(;bt<end;++bt)buf[bt]=buf[bt-dt]}}st.l=lm,st.p=lpos,st.b=bt,st.f=final,lm&&(final=1,st.m=lbt,st.d=dm,st.n=dbt)}while(!final);return bt!=buf.length&&noBuf?slc(buf,0,bt):buf.subarray(0,bt)},"inflt");var et=new u8(0);var zls=__name(function(d,dict){return((d[0]&15)!=8||d[0]>>4>7||(d[0]<<8|d[1])%31)&&err(6,"invalid zlib data"),(d[1]>>5&1)==+!dict&&err(6,"invalid zlib data: "+(d[1]&32?"need":"unexpected")+" dictionary"),(d[1]>>3&4)+2},"zls");function unzlibSync(data,opts){return inflt(data.subarray(zls(data,opts&&opts.dictionary),-4),{i:2},opts&&opts.out,opts&&opts.dictionary)}__name(unzlibSync,"unzlibSync");var td=typeof TextDecoder<"u"&&new TextDecoder,tds=0;try{td.decode(et,{stream:!0}),tds=1}catch{}function findSpan(p,u,U){const n=U.length-p-1;if(u>=U[n])return n-1;if(u<=U[p])return p;let low=p,high=n,mid=Math.floor((low+high)/2);for(;u<U[mid]||u>=U[mid+1];)u<U[mid]?high=mid:low=mid,mid=Math.floor((low+high)/2);return mid}__name(findSpan,"findSpan");function calcBasisFunctions(span,u,p,U){const N=[],left=[],right=[];N[0]=1;for(let j=1;j<=p;++j){left[j]=u-U[span+1-j],right[j]=U[span+j]-u;let saved=0;for(let r=0;r<j;++r){const rv=right[r+1],lv=left[j-r],temp=N[r]/(rv+lv);N[r]=saved+rv*temp,saved=lv*temp}N[j]=saved}return N}__name(calcBasisFunctions,"calcBasisFunctions");function calcBSplinePoint(p,U,P,u){const span=findSpan(p,u,U),N=calcBasisFunctions(span,u,p,U),C=new Vector4(0,0,0,0);for(let j=0;j<=p;++j){const point=P[span-p+j],Nj=N[j],wNj=point.w*Nj;C.x+=point.x*wNj,C.y+=point.y*wNj,C.z+=point.z*wNj,C.w+=point.w*Nj}return C}__name(calcBSplinePoint,"calcBSplinePoint");function calcBasisFunctionDerivatives(span,u,p,n,U){const zeroArr=[];for(let i=0;i<=p;++i)zeroArr[i]=0;const ders=[];for(let i=0;i<=n;++i)ders[i]=zeroArr.slice(0);const ndu=[];for(let i=0;i<=p;++i)ndu[i]=zeroArr.slice(0);ndu[0][0]=1;const left=zeroArr.slice(0),right=zeroArr.slice(0);for(let j=1;j<=p;++j){left[j]=u-U[span+1-j],right[j]=U[span+j]-u;let saved=0;for(let r2=0;r2<j;++r2){const rv=right[r2+1],lv=left[j-r2];ndu[j][r2]=rv+lv;const temp=ndu[r2][j-1]/ndu[j][r2];ndu[r2][j]=saved+rv*temp,saved=lv*temp}ndu[j][j]=saved}for(let j=0;j<=p;++j)ders[0][j]=ndu[j][p];for(let r2=0;r2<=p;++r2){let s1=0,s2=1;const a=[];for(let i=0;i<=p;++i)a[i]=zeroArr.slice(0);a[0][0]=1;for(let k=1;k<=n;++k){let d=0;const rk=r2-k,pk=p-k;r2>=k&&(a[s2][0]=a[s1][0]/ndu[pk+1][rk],d=a[s2][0]*ndu[rk][pk]);const j1=rk>=-1?1:-rk,j2=r2-1<=pk?k-1:p-r2;for(let j3=j1;j3<=j2;++j3)a[s2][j3]=(a[s1][j3]-a[s1][j3-1])/ndu[pk+1][rk+j3],d+=a[s2][j3]*ndu[rk+j3][pk];r2<=pk&&(a[s2][k]=-a[s1][k-1]/ndu[pk+1][r2],d+=a[s2][k]*ndu[r2][pk]),ders[k][r2]=d;const j=s1;s1=s2,s2=j}}let r=p;for(let k=1;k<=n;++k){for(let j=0;j<=p;++j)ders[k][j]*=r;r*=p-k}return ders}__name(calcBasisFunctionDerivatives,"calcBasisFunctionDerivatives");function calcBSplineDerivatives(p,U,P,u,nd){const du=nd<p?nd:p,CK=[],span=findSpan(p,u,U),nders=calcBasisFunctionDerivatives(span,u,p,du,U),Pw=[];for(let i=0;i<P.length;++i){const point=P[i].clone(),w=point.w;point.x*=w,point.y*=w,point.z*=w,Pw[i]=point}for(let k=0;k<=du;++k){const point=Pw[span-p].clone().multiplyScalar(nders[k][0]);for(let j=1;j<=p;++j)point.add(Pw[span-p+j].clone().multiplyScalar(nders[k][j]));CK[k]=point}for(let k=du+1;k<=nd+1;++k)CK[k]=new Vector4(0,0,0);return CK}__name(calcBSplineDerivatives,"calcBSplineDerivatives");function calcKoverI(k,i){let nom=1;for(let j=2;j<=k;++j)nom*=j;let denom=1;for(let j=2;j<=i;++j)denom*=j;for(let j=2;j<=k-i;++j)denom*=j;return nom/denom}__name(calcKoverI,"calcKoverI");function calcRationalCurveDerivatives(Pders){const nd=Pders.length,Aders=[],wders=[];for(let i=0;i<nd;++i){const point=Pders[i];Aders[i]=new Vector3(point.x,point.y,point.z),wders[i]=point.w}const CK=[];for(let k=0;k<nd;++k){const v=Aders[k].clone();for(let i=1;i<=k;++i)v.sub(CK[k-i].clone().multiplyScalar(calcKoverI(k,i)*wders[i]));CK[k]=v.divideScalar(wders[0])}return CK}__name(calcRationalCurveDerivatives,"calcRationalCurveDerivatives");function calcNURBSDerivatives(p,U,P,u,nd){const Pders=calcBSplineDerivatives(p,U,P,u,nd);return calcRationalCurveDerivatives(Pders)}__name(calcNURBSDerivatives,"calcNURBSDerivatives");class NURBSCurve extends Curve{static{__name(this,"NURBSCurve")}constructor(degree,knots,controlPoints,startKnot,endKnot){super();const knotsLength=knots?knots.length-1:0,pointsLength=controlPoints?controlPoints.length:0;this.degree=degree,this.knots=knots,this.controlPoints=[],this.startKnot=startKnot||0,this.endKnot=endKnot||knotsLength;for(let i=0;i<pointsLength;++i){const point=controlPoints[i];this.controlPoints[i]=new Vector4(point.x,point.y,point.z,point.w)}}getPoint(t2,optionalTarget=new Vector3){const point=optionalTarget,u=this.knots[this.startKnot]+t2*(this.knots[this.endKnot]-this.knots[this.startKnot]),hpoint=calcBSplinePoint(this.degree,this.knots,this.controlPoints,u);return hpoint.w!==1&&hpoint.divideScalar(hpoint.w),point.set(hpoint.x,hpoint.y,hpoint.z)}getTangent(t2,optionalTarget=new Vector3){const tangent=optionalTarget,u=this.knots[0]+t2*(this.knots[this.knots.length-1]-this.knots[0]),ders=calcNURBSDerivatives(this.degree,this.knots,this.controlPoints,u,1);return tangent.copy(ders[1]).normalize(),tangent}toJSON(){const data=super.toJSON();return data.degree=this.degree,data.knots=[...this.knots],data.controlPoints=this.controlPoints.map(p=>p.toArray()),data.startKnot=this.startKnot,data.endKnot=this.endKnot,data}fromJSON(json){return super.fromJSON(json),this.degree=json.degree,this.knots=[...json.knots],this.controlPoints=json.controlPoints.map(p=>new Vector4(p[0],p[1],p[2],p[3])),this.startKnot=json.startKnot,this.endKnot=json.endKnot,this}}let fbxTree,connections,sceneGraph;class FBXLoader extends Loader{static{__name(this,"FBXLoader")}constructor(manager){super(manager)}load(url,onLoad,onProgress,onError){const scope=this,path=scope.path===""?LoaderUtils.extractUrlBase(url):scope.path,loader=new FileLoader(this.manager);loader.setPath(scope.path),loader.setResponseType("arraybuffer"),loader.setRequestHeader(scope.requestHeader),loader.setWithCredentials(scope.withCredentials),loader.load(url,function(buffer){try{onLoad(scope.parse(buffer,path))}catch(e){onError?onError(e):console.error(e),scope.manager.itemError(url)}},onProgress,onError)}parse(FBXBuffer,path){if(isFbxFormatBinary(FBXBuffer))fbxTree=new BinaryParser().parse(FBXBuffer);else{const FBXText=convertArrayBufferToString(FBXBuffer);if(!isFbxFormatASCII(FBXText))throw new Error("THREE.FBXLoader: Unknown format.");if(getFbxVersion(FBXText)<7e3)throw new Error("THREE.FBXLoader: FBX version not supported, FileVersion: "+getFbxVersion(FBXText));fbxTree=new TextParser().parse(FBXText)}const textureLoader=new TextureLoader(this.manager).setPath(this.resourcePath||path).setCrossOrigin(this.crossOrigin);return new FBXTreeParser(textureLoader,this.manager).parse(fbxTree)}}class FBXTreeParser{static{__name(this,"FBXTreeParser")}constructor(textureLoader,manager){this.textureLoader=textureLoader,this.manager=manager}parse(){connections=this.parseConnections();const images=this.parseImages(),textures=this.parseTextures(images),materials=this.parseMaterials(textures),deformers=this.parseDeformers(),geometryMap=new GeometryParser().parse(deformers);return this.parseScene(deformers,geometryMap,materials),sceneGraph}parseConnections(){const connectionMap=new Map;return"Connections"in fbxTree&&fbxTree.Connections.connections.forEach(function(rawConnection){const fromID=rawConnection[0],toID=rawConnection[1],relationship=rawConnection[2];connectionMap.has(fromID)||connectionMap.set(fromID,{parents:[],children:[]});const parentRelationship={ID:toID,relationship};connectionMap.get(fromID).parents.push(parentRelationship),connectionMap.has(toID)||connectionMap.set(toID,{parents:[],children:[]});const childRelationship={ID:fromID,relationship};connectionMap.get(toID).children.push(childRelationship)}),connectionMap}parseImages(){const images={},blobs={};if("Video"in fbxTree.Objects){const videoNodes=fbxTree.Objects.Video;for(const nodeID in videoNodes){const videoNode=videoNodes[nodeID],id2=parseInt(nodeID);if(images[id2]=videoNode.RelativeFilename||videoNode.Filename,"Content"in videoNode){const arrayBufferContent=videoNode.Content instanceof ArrayBuffer&&videoNode.Content.byteLength>0,base64Content=typeof videoNode.Content=="string"&&videoNode.Content!=="";if(arrayBufferContent||base64Content){const image=this.parseImage(videoNodes[nodeID]);blobs[videoNode.RelativeFilename||videoNode.Filename]=image}}}}for(const id2 in images){const filename=images[id2];blobs[filename]!==void 0?images[id2]=blobs[filename]:images[id2]=images[id2].split("\\").pop()}return images}parseImage(videoNode){const content=videoNode.Content,fileName=videoNode.RelativeFilename||videoNode.Filename,extension=fileName.slice(fileName.lastIndexOf(".")+1).toLowerCase();let type;switch(extension){case"bmp":type="image/bmp";break;case"jpg":case"jpeg":type="image/jpeg";break;case"png":type="image/png";break;case"tif":type="image/tiff";break;case"tga":this.manager.getHandler(".tga")===null&&console.warn("FBXLoader: TGA loader not found, skipping ",fileName),type="image/tga";break;default:console.warn('FBXLoader: Image type "'+extension+'" is not supported.');return}if(typeof content=="string")return"data:"+type+";base64,"+content;{const array=new Uint8Array(content);return window.URL.createObjectURL(new Blob([array],{type}))}}parseTextures(images){const textureMap=new Map;if("Texture"in fbxTree.Objects){const textureNodes=fbxTree.Objects.Texture;for(const nodeID in textureNodes){const texture=this.parseTexture(textureNodes[nodeID],images);textureMap.set(parseInt(nodeID),texture)}}return textureMap}parseTexture(textureNode,images){const texture=this.loadTexture(textureNode,images);texture.ID=textureNode.id,texture.name=textureNode.attrName;const wrapModeU=textureNode.WrapModeU,wrapModeV=textureNode.WrapModeV,valueU=wrapModeU!==void 0?wrapModeU.value:0,valueV=wrapModeV!==void 0?wrapModeV.value:0;if(texture.wrapS=valueU===0?RepeatWrapping:ClampToEdgeWrapping,texture.wrapT=valueV===0?RepeatWrapping:ClampToEdgeWrapping,"Scaling"in textureNode){const values=textureNode.Scaling.value;texture.repeat.x=values[0],texture.repeat.y=values[1]}if("Translation"in textureNode){const values=textureNode.Translation.value;texture.offset.x=values[0],texture.offset.y=values[1]}return texture}loadTexture(textureNode,images){const nonNativeExtensions=new Set(["tga","tif","tiff","exr","dds","hdr","ktx2"]),extension=textureNode.FileName.split(".").pop().toLowerCase(),loader=nonNativeExtensions.has(extension)?this.manager.getHandler(`.${extension}`):this.textureLoader;if(!loader)return console.warn(`FBXLoader: ${extension.toUpperCase()} loader not found, creating placeholder texture for`,textureNode.RelativeFilename),new Texture;const loaderPath=loader.path;loaderPath||loader.setPath(this.textureLoader.path);const children=connections.get(textureNode.id).children;let fileName;children!==void 0&&children.length>0&&images[children[0].ID]!==void 0&&(fileName=images[children[0].ID],(fileName.indexOf("blob:")===0||fileName.indexOf("data:")===0)&&loader.setPath(void 0));const texture=loader.load(fileName);return loader.setPath(loaderPath),texture}parseMaterials(textureMap){const materialMap=new Map;if("Material"in fbxTree.Objects){const materialNodes=fbxTree.Objects.Material;for(const nodeID in materialNodes){const material=this.parseMaterial(materialNodes[nodeID],textureMap);material!==null&&materialMap.set(parseInt(nodeID),material)}}return materialMap}parseMaterial(materialNode,textureMap){const ID=materialNode.id,name=materialNode.attrName;let type=materialNode.ShadingModel;if(typeof type=="object"&&(type=type.value),!connections.has(ID))return null;const parameters=this.parseParameters(materialNode,textureMap,ID);let material;switch(type.toLowerCase()){case"phong":material=new MeshPhongMaterial;break;case"lambert":material=new MeshLambertMaterial;break;default:console.warn('THREE.FBXLoader: unknown material type "%s". Defaulting to MeshPhongMaterial.',type),material=new MeshPhongMaterial;break}return material.setValues(parameters),material.name=name,material}parseParameters(materialNode,textureMap,ID){const parameters={};materialNode.BumpFactor&&(parameters.bumpScale=materialNode.BumpFactor.value),materialNode.Diffuse?parameters.color=ColorManagement.toWorkingColorSpace(new Color().fromArray(materialNode.Diffuse.value),SRGBColorSpace):materialNode.DiffuseColor&&(materialNode.DiffuseColor.type==="Color"||materialNode.DiffuseColor.type==="ColorRGB")&&(parameters.color=ColorManagement.toWorkingColorSpace(new Color().fromArray(materialNode.DiffuseColor.value),SRGBColorSpace)),materialNode.DisplacementFactor&&(parameters.displacementScale=materialNode.DisplacementFactor.value),materialNode.Emissive?parameters.emissive=ColorManagement.toWorkingColorSpace(new Color().fromArray(materialNode.Emissive.value),SRGBColorSpace):materialNode.EmissiveColor&&(materialNode.EmissiveColor.type==="Color"||materialNode.EmissiveColor.type==="ColorRGB")&&(parameters.emissive=ColorManagement.toWorkingColorSpace(new Color().fromArray(materialNode.EmissiveColor.value),SRGBColorSpace)),materialNode.EmissiveFactor&&(parameters.emissiveIntensity=parseFloat(materialNode.EmissiveFactor.value)),parameters.opacity=1-(materialNode.TransparencyFactor?parseFloat(materialNode.TransparencyFactor.value):0),(parameters.opacity===1||parameters.opacity===0)&&(parameters.opacity=materialNode.Opacity?parseFloat(materialNode.Opacity.value):null,parameters.opacity===null&&(parameters.opacity=1-(materialNode.TransparentColor?parseFloat(materialNode.TransparentColor.value[0]):0))),parameters.opacity<1&&(parameters.transparent=!0),materialNode.ReflectionFactor&&(parameters.reflectivity=materialNode.ReflectionFactor.value),materialNode.Shininess&&(parameters.shininess=materialNode.Shininess.value),materialNode.Specular?parameters.specular=ColorManagement.toWorkingColorSpace(new Color().fromArray(materialNode.Specular.value),SRGBColorSpace):materialNode.SpecularColor&&materialNode.SpecularColor.type==="Color"&&(parameters.specular=ColorManagement.toWorkingColorSpace(new Color().fromArray(materialNode.SpecularColor.value),SRGBColorSpace));const scope=this;return connections.get(ID).children.forEach(function(child){const type=child.relationship;switch(type){case"Bump":parameters.bumpMap=scope.getTexture(textureMap,child.ID);break;case"Maya|TEX_ao_map":parameters.aoMap=scope.getTexture(textureMap,child.ID);break;case"DiffuseColor":case"Maya|TEX_color_map":parameters.map=scope.getTexture(textureMap,child.ID),parameters.map!==void 0&&(parameters.map.colorSpace=SRGBColorSpace);break;case"DisplacementColor":parameters.displacementMap=scope.getTexture(textureMap,child.ID);break;case"EmissiveColor":parameters.emissiveMap=scope.getTexture(textureMap,child.ID),parameters.emissiveMap!==void 0&&(parameters.emissiveMap.colorSpace=SRGBColorSpace);break;case"NormalMap":case"Maya|TEX_normal_map":parameters.normalMap=scope.getTexture(textureMap,child.ID);break;case"ReflectionColor":parameters.envMap=scope.getTexture(textureMap,child.ID),parameters.envMap!==void 0&&(parameters.envMap.mapping=EquirectangularReflectionMapping,parameters.envMap.colorSpace=SRGBColorSpace);break;case"SpecularColor":parameters.specularMap=scope.getTexture(textureMap,child.ID),parameters.specularMap!==void 0&&(parameters.specularMap.colorSpace=SRGBColorSpace);break;case"TransparentColor":case"TransparencyFactor":parameters.alphaMap=scope.getTexture(textureMap,child.ID),parameters.transparent=!0;break;case"AmbientColor":case"ShininessExponent":case"SpecularFactor":case"VectorDisplacementColor":default:console.warn("THREE.FBXLoader: %s map is not supported in three.js, skipping texture.",type);break}}),parameters}getTexture(textureMap,id2){return"LayeredTexture"in fbxTree.Objects&&id2 in fbxTree.Objects.LayeredTexture&&(console.warn("THREE.FBXLoader: layered textures are not supported in three.js. Discarding all but first layer."),id2=connections.get(id2).children[0].ID),textureMap.get(id2)}parseDeformers(){const skeletons={},morphTargets={};if("Deformer"in fbxTree.Objects){const DeformerNodes=fbxTree.Objects.Deformer;for(const nodeID in DeformerNodes){const deformerNode=DeformerNodes[nodeID],relationships=connections.get(parseInt(nodeID));if(deformerNode.attrType==="Skin"){const skeleton=this.parseSkeleton(relationships,DeformerNodes);skeleton.ID=nodeID,relationships.parents.length>1&&console.warn("THREE.FBXLoader: skeleton attached to more than one geometry is not supported."),skeleton.geometryID=relationships.parents[0].ID,skeletons[nodeID]=skeleton}else if(deformerNode.attrType==="BlendShape"){const morphTarget={id:nodeID};morphTarget.rawTargets=this.parseMorphTargets(relationships,DeformerNodes),morphTarget.id=nodeID,relationships.parents.length>1&&console.warn("THREE.FBXLoader: morph target attached to more than one geometry is not supported."),morphTargets[nodeID]=morphTarget}}}return{skeletons,morphTargets}}parseSkeleton(relationships,deformerNodes){const rawBones=[];return relationships.children.forEach(function(child){const boneNode=deformerNodes[child.ID];if(boneNode.attrType!=="Cluster")return;const rawBone={ID:child.ID,indices:[],weights:[],transformLink:new Matrix4().fromArray(boneNode.TransformLink.a)};"Indexes"in boneNode&&(rawBone.indices=boneNode.Indexes.a,rawBone.weights=boneNode.Weights.a),rawBones.push(rawBone)}),{rawBones,bones:[]}}parseMorphTargets(relationships,deformerNodes){const rawMorphTargets=[];for(let i=0;i<relationships.children.length;i++){const child=relationships.children[i],morphTargetNode=deformerNodes[child.ID],rawMorphTarget={name:morphTargetNode.attrName,initialWeight:morphTargetNode.DeformPercent,id:morphTargetNode.id,fullWeights:morphTargetNode.FullWeights.a};if(morphTargetNode.attrType!=="BlendShapeChannel")return;rawMorphTarget.geoID=connections.get(parseInt(child.ID)).children.filter(function(child2){return child2.relationship===void 0})[0].ID,rawMorphTargets.push(rawMorphTarget)}return rawMorphTargets}parseScene(deformers,geometryMap,materialMap){sceneGraph=new Group;const modelMap=this.parseModels(deformers.skeletons,geometryMap,materialMap),modelNodes=fbxTree.Objects.Model,scope=this;modelMap.forEach(function(model){const modelNode=modelNodes[model.ID];scope.setLookAtProperties(model,modelNode),connections.get(model.ID).parents.forEach(function(connection){const parent=modelMap.get(connection.ID);parent!==void 0&&parent.add(model)}),model.parent===null&&sceneGraph.add(model)}),this.bindSkeleton(deformers.skeletons,geometryMap,modelMap),this.addGlobalSceneSettings(),sceneGraph.traverse(function(node){if(node.userData.transformData){node.parent&&(node.userData.transformData.parentMatrix=node.parent.matrix,node.userData.transformData.parentMatrixWorld=node.parent.matrixWorld);const transform=generateTransform(node.userData.transformData);node.applyMatrix4(transform),node.updateWorldMatrix()}});const animations=new AnimationParser().parse();sceneGraph.children.length===1&&sceneGraph.children[0].isGroup&&(sceneGraph.children[0].animations=animations,sceneGraph=sceneGraph.children[0]),sceneGraph.animations=animations}parseModels(skeletons,geometryMap,materialMap){const modelMap=new Map,modelNodes=fbxTree.Objects.Model;for(const nodeID in modelNodes){const id2=parseInt(nodeID),node=modelNodes[nodeID],relationships=connections.get(id2);let model=this.buildSkeleton(relationships,skeletons,id2,node.attrName);if(!model){switch(node.attrType){case"Camera":model=this.createCamera(relationships);break;case"Light":model=this.createLight(relationships);break;case"Mesh":model=this.createMesh(relationships,geometryMap,materialMap);break;case"NurbsCurve":model=this.createCurve(relationships,geometryMap);break;case"LimbNode":case"Root":model=new Bone;break;case"Null":default:model=new Group;break}model.name=node.attrName?PropertyBinding.sanitizeNodeName(node.attrName):"",model.userData.originalName=node.attrName,model.ID=id2}this.getTransformData(model,node),modelMap.set(id2,model)}return modelMap}buildSkeleton(relationships,skeletons,id2,name){let bone=null;return relationships.parents.forEach(function(parent){for(const ID in skeletons){const skeleton=skeletons[ID];skeleton.rawBones.forEach(function(rawBone,i){if(rawBone.ID===parent.ID){const subBone=bone;bone=new Bone,bone.matrixWorld.copy(rawBone.transformLink),bone.name=name?PropertyBinding.sanitizeNodeName(name):"",bone.userData.originalName=name,bone.ID=id2,skeleton.bones[i]=bone,subBone!==null&&bone.add(subBone)}})}}),bone}createCamera(relationships){let model,cameraAttribute;if(relationships.children.forEach(function(child){const attr=fbxTree.Objects.NodeAttribute[child.ID];attr!==void 0&&(cameraAttribute=attr)}),cameraAttribute===void 0)model=new Object3D;else{let type=0;cameraAttribute.CameraProjectionType!==void 0&&cameraAttribute.CameraProjectionType.value===1&&(type=1);let nearClippingPlane=1;cameraAttribute.NearPlane!==void 0&&(nearClippingPlane=cameraAttribute.NearPlane.value/1e3);let farClippingPlane=1e3;cameraAttribute.FarPlane!==void 0&&(farClippingPlane=cameraAttribute.FarPlane.value/1e3);let width=window.innerWidth,height=window.innerHeight;cameraAttribute.AspectWidth!==void 0&&cameraAttribute.AspectHeight!==void 0&&(width=cameraAttribute.AspectWidth.value,height=cameraAttribute.AspectHeight.value);const aspect2=width/height;let fov2=45;cameraAttribute.FieldOfView!==void 0&&(fov2=cameraAttribute.FieldOfView.value);const focalLength=cameraAttribute.FocalLength?cameraAttribute.FocalLength.value:null;switch(type){case 0:model=new PerspectiveCamera(fov2,aspect2,nearClippingPlane,farClippingPlane),focalLength!==null&&model.setFocalLength(focalLength);break;case 1:console.warn("THREE.FBXLoader: Orthographic cameras not supported yet."),model=new Object3D;break;default:console.warn("THREE.FBXLoader: Unknown camera type "+type+"."),model=new Object3D;break}}return model}createLight(relationships){let model,lightAttribute;if(relationships.children.forEach(function(child){const attr=fbxTree.Objects.NodeAttribute[child.ID];attr!==void 0&&(lightAttribute=attr)}),lightAttribute===void 0)model=new Object3D;else{let type;lightAttribute.LightType===void 0?type=0:type=lightAttribute.LightType.value;let color=16777215;lightAttribute.Color!==void 0&&(color=ColorManagement.toWorkingColorSpace(new Color().fromArray(lightAttribute.Color.value),SRGBColorSpace));let intensity=lightAttribute.Intensity===void 0?1:lightAttribute.Intensity.value/100;lightAttribute.CastLightOnObject!==void 0&&lightAttribute.CastLightOnObject.value===0&&(intensity=0);let distance=0;lightAttribute.FarAttenuationEnd!==void 0&&(lightAttribute.EnableFarAttenuation!==void 0&&lightAttribute.EnableFarAttenuation.value===0?distance=0:distance=lightAttribute.FarAttenuationEnd.value);const decay=1;switch(type){case 0:model=new PointLight(color,intensity,distance,decay);break;case 1:model=new DirectionalLight(color,intensity);break;case 2:let angle=Math.PI/3;lightAttribute.InnerAngle!==void 0&&(angle=MathUtils.degToRad(lightAttribute.InnerAngle.value));let penumbra=0;lightAttribute.OuterAngle!==void 0&&(penumbra=MathUtils.degToRad(lightAttribute.OuterAngle.value),penumbra=Math.max(penumbra,1)),model=new SpotLight(color,intensity,distance,angle,penumbra,decay);break;default:console.warn("THREE.FBXLoader: Unknown light type "+lightAttribute.LightType.value+", defaulting to a PointLight."),model=new PointLight(color,intensity);break}lightAttribute.CastShadows!==void 0&&lightAttribute.CastShadows.value===1&&(model.castShadow=!0)}return model}createMesh(relationships,geometryMap,materialMap){let model,geometry=null,material=null;const materials=[];return relationships.children.forEach(function(child){geometryMap.has(child.ID)&&(geometry=geometryMap.get(child.ID)),materialMap.has(child.ID)&&materials.push(materialMap.get(child.ID))}),materials.length>1?material=materials:materials.length>0?material=materials[0]:(material=new MeshPhongMaterial({name:Loader.DEFAULT_MATERIAL_NAME,color:13421772}),materials.push(material)),"color"in geometry.attributes&&materials.forEach(function(material2){material2.vertexColors=!0}),geometry.FBX_Deformer?(model=new SkinnedMesh(geometry,material),model.normalizeSkinWeights()):model=new Mesh(geometry,material),model}createCurve(relationships,geometryMap){const geometry=relationships.children.reduce(function(geo,child){return geometryMap.has(child.ID)&&(geo=geometryMap.get(child.ID)),geo},null),material=new LineBasicMaterial({name:Loader.DEFAULT_MATERIAL_NAME,color:3342591,linewidth:1});return new Line(geometry,material)}getTransformData(model,modelNode){const transformData={};"InheritType"in modelNode&&(transformData.inheritType=parseInt(modelNode.InheritType.value)),"RotationOrder"in modelNode?transformData.eulerOrder=getEulerOrder(modelNode.RotationOrder.value):transformData.eulerOrder=getEulerOrder(0),"Lcl_Translation"in modelNode&&(transformData.translation=modelNode.Lcl_Translation.value),"PreRotation"in modelNode&&(transformData.preRotation=modelNode.PreRotation.value),"Lcl_Rotation"in modelNode&&(transformData.rotation=modelNode.Lcl_Rotation.value),"PostRotation"in modelNode&&(transformData.postRotation=modelNode.PostRotation.value),"Lcl_Scaling"in modelNode&&(transformData.scale=modelNode.Lcl_Scaling.value),"ScalingOffset"in modelNode&&(transformData.scalingOffset=modelNode.ScalingOffset.value),"ScalingPivot"in modelNode&&(transformData.scalingPivot=modelNode.ScalingPivot.value),"RotationOffset"in modelNode&&(transformData.rotationOffset=modelNode.RotationOffset.value),"RotationPivot"in modelNode&&(transformData.rotationPivot=modelNode.RotationPivot.value),model.userData.transformData=transformData}setLookAtProperties(model,modelNode){"LookAtProperty"in modelNode&&connections.get(model.ID).children.forEach(function(child){if(child.relationship==="LookAtProperty"){const lookAtTarget=fbxTree.Objects.Model[child.ID];if("Lcl_Translation"in lookAtTarget){const pos=lookAtTarget.Lcl_Translation.value;model.target!==void 0?(model.target.position.fromArray(pos),sceneGraph.add(model.target)):model.lookAt(new Vector3().fromArray(pos))}}})}bindSkeleton(skeletons,geometryMap,modelMap){const bindMatrices=this.parsePoseNodes();for(const ID in skeletons){const skeleton=skeletons[ID];connections.get(parseInt(skeleton.ID)).parents.forEach(function(parent){if(geometryMap.has(parent.ID)){const geoID=parent.ID;connections.get(geoID).parents.forEach(function(geoConnParent){modelMap.has(geoConnParent.ID)&&modelMap.get(geoConnParent.ID).bind(new Skeleton(skeleton.bones),bindMatrices[geoConnParent.ID])})}})}}parsePoseNodes(){const bindMatrices={};if("Pose"in fbxTree.Objects){const BindPoseNode=fbxTree.Objects.Pose;for(const nodeID in BindPoseNode)if(BindPoseNode[nodeID].attrType==="BindPose"&&BindPoseNode[nodeID].NbPoseNodes>0){const poseNodes=BindPoseNode[nodeID].PoseNode;Array.isArray(poseNodes)?poseNodes.forEach(function(poseNode){bindMatrices[poseNode.Node]=new Matrix4().fromArray(poseNode.Matrix.a)}):bindMatrices[poseNodes.Node]=new Matrix4().fromArray(poseNodes.Matrix.a)}}return bindMatrices}addGlobalSceneSettings(){if("GlobalSettings"in fbxTree){if("AmbientColor"in fbxTree.GlobalSettings){const ambientColor=fbxTree.GlobalSettings.AmbientColor.value,r=ambientColor[0],g=ambientColor[1],b=ambientColor[2];if(r!==0||g!==0||b!==0){const color=new Color().setRGB(r,g,b,SRGBColorSpace);sceneGraph.add(new AmbientLight(color,1))}}"UnitScaleFactor"in fbxTree.GlobalSettings&&(sceneGraph.userData.unitScaleFactor=fbxTree.GlobalSettings.UnitScaleFactor.value)}}}class GeometryParser{static{__name(this,"GeometryParser")}constructor(){this.negativeMaterialIndices=!1}parse(deformers){const geometryMap=new Map;if("Geometry"in fbxTree.Objects){const geoNodes=fbxTree.Objects.Geometry;for(const nodeID in geoNodes){const relationships=connections.get(parseInt(nodeID)),geo=this.parseGeometry(relationships,geoNodes[nodeID],deformers);geometryMap.set(parseInt(nodeID),geo)}}return this.negativeMaterialIndices===!0&&console.warn("THREE.FBXLoader: The FBX file contains invalid (negative) material indices. The asset might not render as expected."),geometryMap}parseGeometry(relationships,geoNode,deformers){switch(geoNode.attrType){case"Mesh":return this.parseMeshGeometry(relationships,geoNode,deformers);case"NurbsCurve":return this.parseNurbsGeometry(geoNode)}}parseMeshGeometry(relationships,geoNode,deformers){const skeletons=deformers.skeletons,morphTargets=[],modelNodes=relationships.parents.map(function(parent){return fbxTree.Objects.Model[parent.ID]});if(modelNodes.length===0)return;const skeleton=relationships.children.reduce(function(skeleton2,child){return skeletons[child.ID]!==void 0&&(skeleton2=skeletons[child.ID]),skeleton2},null);relationships.children.forEach(function(child){deformers.morphTargets[child.ID]!==void 0&&morphTargets.push(deformers.morphTargets[child.ID])});const modelNode=modelNodes[0],transformData={};"RotationOrder"in modelNode&&(transformData.eulerOrder=getEulerOrder(modelNode.RotationOrder.value)),"InheritType"in modelNode&&(transformData.inheritType=parseInt(modelNode.InheritType.value)),"GeometricTranslation"in modelNode&&(transformData.translation=modelNode.GeometricTranslation.value),"GeometricRotation"in modelNode&&(transformData.rotation=modelNode.GeometricRotation.value),"GeometricScaling"in modelNode&&(transformData.scale=modelNode.GeometricScaling.value);const transform=generateTransform(transformData);return this.genGeometry(geoNode,skeleton,morphTargets,transform)}genGeometry(geoNode,skeleton,morphTargets,preTransform){const geo=new BufferGeometry;geoNode.attrName&&(geo.name=geoNode.attrName);const geoInfo=this.parseGeoNode(geoNode,skeleton),buffers=this.genBuffers(geoInfo),positionAttribute=new Float32BufferAttribute(buffers.vertex,3);if(positionAttribute.applyMatrix4(preTransform),geo.setAttribute("position",positionAttribute),buffers.colors.length>0&&geo.setAttribute("color",new Float32BufferAttribute(buffers.colors,3)),skeleton&&(geo.setAttribute("skinIndex",new Uint16BufferAttribute(buffers.weightsIndices,4)),geo.setAttribute("skinWeight",new Float32BufferAttribute(buffers.vertexWeights,4)),geo.FBX_Deformer=skeleton),buffers.normal.length>0){const normalMatrix=new Matrix3().getNormalMatrix(preTransform),normalAttribute=new Float32BufferAttribute(buffers.normal,3);normalAttribute.applyNormalMatrix(normalMatrix),geo.setAttribute("normal",normalAttribute)}if(buffers.uvs.forEach(function(uvBuffer,i){const name=i===0?"uv":`uv${i}`;geo.setAttribute(name,new Float32BufferAttribute(buffers.uvs[i],2))}),geoInfo.material&&geoInfo.material.mappingType!=="AllSame"){let prevMaterialIndex=buffers.materialIndex[0],startIndex=0;if(buffers.materialIndex.forEach(function(currentIndex,i){currentIndex!==prevMaterialIndex&&(geo.addGroup(startIndex,i-startIndex,prevMaterialIndex),prevMaterialIndex=currentIndex,startIndex=i)}),geo.groups.length>0){const lastGroup=geo.groups[geo.groups.length-1],lastIndex=lastGroup.start+lastGroup.count;lastIndex!==buffers.materialIndex.length&&geo.addGroup(lastIndex,buffers.materialIndex.length-lastIndex,prevMaterialIndex)}geo.groups.length===0&&geo.addGroup(0,buffers.materialIndex.length,buffers.materialIndex[0])}return this.addMorphTargets(geo,geoNode,morphTargets,preTransform),geo}parseGeoNode(geoNode,skeleton){const geoInfo={};if(geoInfo.vertexPositions=geoNode.Vertices!==void 0?geoNode.Vertices.a:[],geoInfo.vertexIndices=geoNode.PolygonVertexIndex!==void 0?geoNode.PolygonVertexIndex.a:[],geoNode.LayerElementColor&&(geoInfo.color=this.parseVertexColors(geoNode.LayerElementColor[0])),geoNode.LayerElementMaterial&&(geoInfo.material=this.parseMaterialIndices(geoNode.LayerElementMaterial[0])),geoNode.LayerElementNormal&&(geoInfo.normal=this.parseNormals(geoNode.LayerElementNormal[0])),geoNode.LayerElementUV){geoInfo.uv=[];let i=0;for(;geoNode.LayerElementUV[i];)geoNode.LayerElementUV[i].UV&&geoInfo.uv.push(this.parseUVs(geoNode.LayerElementUV[i])),i++}return geoInfo.weightTable={},skeleton!==null&&(geoInfo.skeleton=skeleton,skeleton.rawBones.forEach(function(rawBone,i){rawBone.indices.forEach(function(index,j){geoInfo.weightTable[index]===void 0&&(geoInfo.weightTable[index]=[]),geoInfo.weightTable[index].push({id:i,weight:rawBone.weights[j]})})})),geoInfo}genBuffers(geoInfo){const buffers={vertex:[],normal:[],colors:[],uvs:[],materialIndex:[],vertexWeights:[],weightsIndices:[]};let polygonIndex=0,faceLength=0,displayedWeightsWarning=!1,facePositionIndexes=[],faceNormals=[],faceColors=[],faceUVs=[],faceWeights=[],faceWeightIndices=[];const scope=this;return geoInfo.vertexIndices.forEach(function(vertexIndex,polygonVertexIndex){let materialIndex,endOfFace=!1;vertexIndex<0&&(vertexIndex=vertexIndex^-1,endOfFace=!0);let weightIndices=[],weights=[];if(facePositionIndexes.push(vertexIndex*3,vertexIndex*3+1,vertexIndex*3+2),geoInfo.color){const data=getData(polygonVertexIndex,polygonIndex,vertexIndex,geoInfo.color);faceColors.push(data[0],data[1],data[2])}if(geoInfo.skeleton){if(geoInfo.weightTable[vertexIndex]!==void 0&&geoInfo.weightTable[vertexIndex].forEach(function(wt){weights.push(wt.weight),weightIndices.push(wt.id)}),weights.length>4){displayedWeightsWarning||(console.warn("THREE.FBXLoader: Vertex has more than 4 skinning weights assigned to vertex. Deleting additional weights."),displayedWeightsWarning=!0);const wIndex=[0,0,0,0],Weight=[0,0,0,0];weights.forEach(function(weight,weightIndex){let currentWeight=weight,currentIndex=weightIndices[weightIndex];Weight.forEach(function(comparedWeight,comparedWeightIndex,comparedWeightArray){if(currentWeight>comparedWeight){comparedWeightArray[comparedWeightIndex]=currentWeight,currentWeight=comparedWeight;const tmp=wIndex[comparedWeightIndex];wIndex[comparedWeightIndex]=currentIndex,currentIndex=tmp}})}),weightIndices=wIndex,weights=Weight}for(;weights.length<4;)weights.push(0),weightIndices.push(0);for(let i=0;i<4;++i)faceWeights.push(weights[i]),faceWeightIndices.push(weightIndices[i])}if(geoInfo.normal){const data=getData(polygonVertexIndex,polygonIndex,vertexIndex,geoInfo.normal);faceNormals.push(data[0],data[1],data[2])}geoInfo.material&&geoInfo.material.mappingType!=="AllSame"&&(materialIndex=getData(polygonVertexIndex,polygonIndex,vertexIndex,geoInfo.material)[0],materialIndex<0&&(scope.negativeMaterialIndices=!0,materialIndex=0)),geoInfo.uv&&geoInfo.uv.forEach(function(uv,i){const data=getData(polygonVertexIndex,polygonIndex,vertexIndex,uv);faceUVs[i]===void 0&&(faceUVs[i]=[]),faceUVs[i].push(data[0]),faceUVs[i].push(data[1])}),faceLength++,endOfFace&&(scope.genFace(buffers,geoInfo,facePositionIndexes,materialIndex,faceNormals,faceColors,faceUVs,faceWeights,faceWeightIndices,faceLength),polygonIndex++,faceLength=0,facePositionIndexes=[],faceNormals=[],faceColors=[],faceUVs=[],faceWeights=[],faceWeightIndices=[])}),buffers}getNormalNewell(vertices){const normal=new Vector3(0,0,0);for(let i=0;i<vertices.length;i++){const current=vertices[i],next=vertices[(i+1)%vertices.length];normal.x+=(current.y-next.y)*(current.z+next.z),normal.y+=(current.z-next.z)*(current.x+next.x),normal.z+=(current.x-next.x)*(current.y+next.y)}return normal.normalize(),normal}getNormalTangentAndBitangent(vertices){const normalVector=this.getNormalNewell(vertices),tangent=(Math.abs(normalVector.z)>.5?new Vector3(0,1,0):new Vector3(0,0,1)).cross(normalVector).normalize(),bitangent=normalVector.clone().cross(tangent).normalize();return{normal:normalVector,tangent,bitangent}}flattenVertex(vertex2,normalTangent,normalBitangent){return new Vector2(vertex2.dot(normalTangent),vertex2.dot(normalBitangent))}genFace(buffers,geoInfo,facePositionIndexes,materialIndex,faceNormals,faceColors,faceUVs,faceWeights,faceWeightIndices,faceLength){let triangles;if(faceLength>3){const vertices=[],positions=geoInfo.baseVertexPositions||geoInfo.vertexPositions;for(let i=0;i<facePositionIndexes.length;i+=3)vertices.push(new Vector3(positions[facePositionIndexes[i]],positions[facePositionIndexes[i+1]],positions[facePositionIndexes[i+2]]));const{tangent,bitangent}=this.getNormalTangentAndBitangent(vertices),triangulationInput=[];for(const vertex2 of vertices)triangulationInput.push(this.flattenVertex(vertex2,tangent,bitangent));triangles=ShapeUtils.triangulateShape(triangulationInput,[])}else triangles=[[0,1,2]];for(const[i0,i1,i2]of triangles)buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i0*3]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i0*3+1]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i0*3+2]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i1*3]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i1*3+1]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i1*3+2]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i2*3]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i2*3+1]]),buffers.vertex.push(geoInfo.vertexPositions[facePositionIndexes[i2*3+2]]),geoInfo.skeleton&&(buffers.vertexWeights.push(faceWeights[i0*4]),buffers.vertexWeights.push(faceWeights[i0*4+1]),buffers.vertexWeights.push(faceWeights[i0*4+2]),buffers.vertexWeights.push(faceWeights[i0*4+3]),buffers.vertexWeights.push(faceWeights[i1*4]),buffers.vertexWeights.push(faceWeights[i1*4+1]),buffers.vertexWeights.push(faceWeights[i1*4+2]),buffers.vertexWeights.push(faceWeights[i1*4+3]),buffers.vertexWeights.push(faceWeights[i2*4]),buffers.vertexWeights.push(faceWeights[i2*4+1]),buffers.vertexWeights.push(faceWeights[i2*4+2]),buffers.vertexWeights.push(faceWeights[i2*4+3]),buffers.weightsIndices.push(faceWeightIndices[i0*4]),buffers.weightsIndices.push(faceWeightIndices[i0*4+1]),buffers.weightsIndices.push(faceWeightIndices[i0*4+2]),buffers.weightsIndices.push(faceWeightIndices[i0*4+3]),buffers.weightsIndices.push(faceWeightIndices[i1*4]),buffers.weightsIndices.push(faceWeightIndices[i1*4+1]),buffers.weightsIndices.push(faceWeightIndices[i1*4+2]),buffers.weightsIndices.push(faceWeightIndices[i1*4+3]),buffers.weightsIndices.push(faceWeightIndices[i2*4]),buffers.weightsIndices.push(faceWeightIndices[i2*4+1]),buffers.weightsIndices.push(faceWeightIndices[i2*4+2]),buffers.weightsIndices.push(faceWeightIndices[i2*4+3])),geoInfo.color&&(buffers.colors.push(faceColors[i0*3]),buffers.colors.push(faceColors[i0*3+1]),buffers.colors.push(faceColors[i0*3+2]),buffers.colors.push(faceColors[i1*3]),buffers.colors.push(faceColors[i1*3+1]),buffers.colors.push(faceColors[i1*3+2]),buffers.colors.push(faceColors[i2*3]),buffers.colors.push(faceColors[i2*3+1]),buffers.colors.push(faceColors[i2*3+2])),geoInfo.material&&geoInfo.material.mappingType!=="AllSame"&&(buffers.materialIndex.push(materialIndex),buffers.materialIndex.push(materialIndex),buffers.materialIndex.push(materialIndex)),geoInfo.normal&&(buffers.normal.push(faceNormals[i0*3]),buffers.normal.push(faceNormals[i0*3+1]),buffers.normal.push(faceNormals[i0*3+2]),buffers.normal.push(faceNormals[i1*3]),buffers.normal.push(faceNormals[i1*3+1]),buffers.normal.push(faceNormals[i1*3+2]),buffers.normal.push(faceNormals[i2*3]),buffers.normal.push(faceNormals[i2*3+1]),buffers.normal.push(faceNormals[i2*3+2])),geoInfo.uv&&geoInfo.uv.forEach(function(uv,j){buffers.uvs[j]===void 0&&(buffers.uvs[j]=[]),buffers.uvs[j].push(faceUVs[j][i0*2]),buffers.uvs[j].push(faceUVs[j][i0*2+1]),buffers.uvs[j].push(faceUVs[j][i1*2]),buffers.uvs[j].push(faceUVs[j][i1*2+1]),buffers.uvs[j].push(faceUVs[j][i2*2]),buffers.uvs[j].push(faceUVs[j][i2*2+1])})}addMorphTargets(parentGeo,parentGeoNode,morphTargets,preTransform){if(morphTargets.length===0)return;parentGeo.morphTargetsRelative=!0,parentGeo.morphAttributes.position=[];const scope=this;morphTargets.forEach(function(morphTarget){morphTarget.rawTargets.forEach(function(rawTarget){const morphGeoNode=fbxTree.Objects.Geometry[rawTarget.geoID];morphGeoNode!==void 0&&scope.genMorphGeometry(parentGeo,parentGeoNode,morphGeoNode,preTransform,rawTarget.name)})})}genMorphGeometry(parentGeo,parentGeoNode,morphGeoNode,preTransform,name){const basePositions=parentGeoNode.Vertices!==void 0?parentGeoNode.Vertices.a:[],baseIndices=parentGeoNode.PolygonVertexIndex!==void 0?parentGeoNode.PolygonVertexIndex.a:[],morphPositionsSparse=morphGeoNode.Vertices!==void 0?morphGeoNode.Vertices.a:[],morphIndices=morphGeoNode.Indexes!==void 0?morphGeoNode.Indexes.a:[],length=parentGeo.attributes.position.count*3,morphPositions=new Float32Array(length);for(let i=0;i<morphIndices.length;i++){const morphIndex=morphIndices[i]*3;morphPositions[morphIndex]=morphPositionsSparse[i*3],morphPositions[morphIndex+1]=morphPositionsSparse[i*3+1],morphPositions[morphIndex+2]=morphPositionsSparse[i*3+2]}const morphGeoInfo={vertexIndices:baseIndices,vertexPositions:morphPositions,baseVertexPositions:basePositions},morphBuffers=this.genBuffers(morphGeoInfo),positionAttribute=new Float32BufferAttribute(morphBuffers.vertex,3);positionAttribute.name=name||morphGeoNode.attrName,positionAttribute.applyMatrix4(preTransform),parentGeo.morphAttributes.position.push(positionAttribute)}parseNormals(NormalNode){const mappingType=NormalNode.MappingInformationType,referenceType=NormalNode.ReferenceInformationType,buffer=NormalNode.Normals.a;let indexBuffer=[];return referenceType==="IndexToDirect"&&("NormalIndex"in NormalNode?indexBuffer=NormalNode.NormalIndex.a:"NormalsIndex"in NormalNode&&(indexBuffer=NormalNode.NormalsIndex.a)),{dataSize:3,buffer,indices:indexBuffer,mappingType,referenceType}}parseUVs(UVNode){const mappingType=UVNode.MappingInformationType,referenceType=UVNode.ReferenceInformationType,buffer=UVNode.UV.a;let indexBuffer=[];return referenceType==="IndexToDirect"&&(indexBuffer=UVNode.UVIndex.a),{dataSize:2,buffer,indices:indexBuffer,mappingType,referenceType}}parseVertexColors(ColorNode){const mappingType=ColorNode.MappingInformationType,referenceType=ColorNode.ReferenceInformationType,buffer=ColorNode.Colors.a;let indexBuffer=[];referenceType==="IndexToDirect"&&(indexBuffer=ColorNode.ColorIndex.a);for(let i=0,c=new Color;i<buffer.length;i+=4)c.fromArray(buffer,i),ColorManagement.toWorkingColorSpace(c,SRGBColorSpace),c.toArray(buffer,i);return{dataSize:4,buffer,indices:indexBuffer,mappingType,referenceType}}parseMaterialIndices(MaterialNode){const mappingType=MaterialNode.MappingInformationType,referenceType=MaterialNode.ReferenceInformationType;if(mappingType==="NoMappingInformation")return{dataSize:1,buffer:[0],indices:[0],mappingType:"AllSame",referenceType};const materialIndexBuffer=MaterialNode.Materials.a,materialIndices=[];for(let i=0;i<materialIndexBuffer.length;++i)materialIndices.push(i);return{dataSize:1,buffer:materialIndexBuffer,indices:materialIndices,mappingType,referenceType}}parseNurbsGeometry(geoNode){const order=parseInt(geoNode.Order);if(isNaN(order))return console.error("THREE.FBXLoader: Invalid Order %s given for geometry ID: %s",geoNode.Order,geoNode.id),new BufferGeometry;const degree=order-1,knots=geoNode.KnotVector.a,controlPoints=[],pointsValues=geoNode.Points.a;for(let i=0,l=pointsValues.length;i<l;i+=4)controlPoints.push(new Vector4().fromArray(pointsValues,i));let startKnot,endKnot;if(geoNode.Form==="Closed")controlPoints.push(controlPoints[0]);else if(geoNode.Form==="Periodic"){startKnot=degree,endKnot=knots.length-1-startKnot;for(let i=0;i<degree;++i)controlPoints.push(controlPoints[i])}const points=new NURBSCurve(degree,knots,controlPoints,startKnot,endKnot).getPoints(controlPoints.length*12);return new BufferGeometry().setFromPoints(points)}}class AnimationParser{static{__name(this,"AnimationParser")}parse(){const animationClips=[],rawClips=this.parseClips();if(rawClips!==void 0)for(const key in rawClips){const rawClip=rawClips[key],clip=this.addClip(rawClip);animationClips.push(clip)}return animationClips}parseClips(){if(fbxTree.Objects.AnimationCurve===void 0)return;const curveNodesMap=this.parseAnimationCurveNodes();this.parseAnimationCurves(curveNodesMap);const layersMap=this.parseAnimationLayers(curveNodesMap);return this.parseAnimStacks(layersMap)}parseAnimationCurveNodes(){const rawCurveNodes=fbxTree.Objects.AnimationCurveNode,curveNodesMap=new Map;for(const nodeID in rawCurveNodes){const rawCurveNode=rawCurveNodes[nodeID];if(rawCurveNode.attrName.match(/S|R|T|DeformPercent/)!==null){const curveNode={id:rawCurveNode.id,attr:rawCurveNode.attrName,curves:{}};curveNodesMap.set(curveNode.id,curveNode)}}return curveNodesMap}parseAnimationCurves(curveNodesMap){const rawCurves=fbxTree.Objects.AnimationCurve;for(const nodeID in rawCurves){const animationCurve={id:rawCurves[nodeID].id,times:rawCurves[nodeID].KeyTime.a.map(convertFBXTimeToSeconds),values:rawCurves[nodeID].KeyValueFloat.a},relationships=connections.get(animationCurve.id);if(relationships!==void 0){const animationCurveID=relationships.parents[0].ID,animationCurveRelationship=relationships.parents[0].relationship;animationCurveRelationship.match(/X/)?curveNodesMap.get(animationCurveID).curves.x=animationCurve:animationCurveRelationship.match(/Y/)?curveNodesMap.get(animationCurveID).curves.y=animationCurve:animationCurveRelationship.match(/Z/)?curveNodesMap.get(animationCurveID).curves.z=animationCurve:animationCurveRelationship.match(/DeformPercent/)&&curveNodesMap.has(animationCurveID)&&(curveNodesMap.get(animationCurveID).curves.morph=animationCurve)}}}parseAnimationLayers(curveNodesMap){const rawLayers=fbxTree.Objects.AnimationLayer,layersMap=new Map;for(const nodeID in rawLayers){const layerCurveNodes=[],connection=connections.get(parseInt(nodeID));connection!==void 0&&(connection.children.forEach(function(child,i){if(curveNodesMap.has(child.ID)){const curveNode=curveNodesMap.get(child.ID);if(curveNode.curves.x!==void 0||curveNode.curves.y!==void 0||curveNode.curves.z!==void 0){if(layerCurveNodes[i]===void 0){const modelID=connections.get(child.ID).parents.filter(function(parent){return parent.relationship!==void 0})[0].ID;if(modelID!==void 0){const rawModel=fbxTree.Objects.Model[modelID.toString()];if(rawModel===void 0){console.warn("THREE.FBXLoader: Encountered a unused curve.",child);return}const node={modelName:rawModel.attrName?PropertyBinding.sanitizeNodeName(rawModel.attrName):"",ID:rawModel.id,initialPosition:[0,0,0],initialRotation:[0,0,0],initialScale:[1,1,1]};sceneGraph.traverse(function(child2){child2.ID===rawModel.id&&(node.transform=child2.matrix,child2.userData.transformData&&(node.eulerOrder=child2.userData.transformData.eulerOrder))}),node.transform||(node.transform=new Matrix4),"PreRotation"in rawModel&&(node.preRotation=rawModel.PreRotation.value),"PostRotation"in rawModel&&(node.postRotation=rawModel.PostRotation.value),layerCurveNodes[i]=node}}layerCurveNodes[i]&&(layerCurveNodes[i][curveNode.attr]=curveNode)}else if(curveNode.curves.morph!==void 0){if(layerCurveNodes[i]===void 0){const deformerID=connections.get(child.ID).parents.filter(function(parent){return parent.relationship!==void 0})[0].ID,morpherID=connections.get(deformerID).parents[0].ID,geoID=connections.get(morpherID).parents[0].ID,modelID=connections.get(geoID).parents[0].ID,rawModel=fbxTree.Objects.Model[modelID],node={modelName:rawModel.attrName?PropertyBinding.sanitizeNodeName(rawModel.attrName):"",morphName:fbxTree.Objects.Deformer[deformerID].attrName};layerCurveNodes[i]=node}layerCurveNodes[i][curveNode.attr]=curveNode}}}),layersMap.set(parseInt(nodeID),layerCurveNodes))}return layersMap}parseAnimStacks(layersMap){const rawStacks=fbxTree.Objects.AnimationStack,rawClips={};for(const nodeID in rawStacks){const children=connections.get(parseInt(nodeID)).children;children.length>1&&console.warn("THREE.FBXLoader: Encountered an animation stack with multiple layers, this is currently not supported. Ignoring subsequent layers.");const layer=layersMap.get(children[0].ID);rawClips[nodeID]={name:rawStacks[nodeID].attrName,layer}}return rawClips}addClip(rawClip){let tracks=[];const scope=this;return rawClip.layer.forEach(function(rawTracks){tracks=tracks.concat(scope.generateTracks(rawTracks))}),new AnimationClip(rawClip.name,-1,tracks)}generateTracks(rawTracks){const tracks=[];let initialPosition=new Vector3,initialScale=new Vector3;if(rawTracks.transform&&rawTracks.transform.decompose(initialPosition,new Quaternion,initialScale),initialPosition=initialPosition.toArray(),initialScale=initialScale.toArray(),rawTracks.T!==void 0&&Object.keys(rawTracks.T.curves).length>0){const positionTrack=this.generateVectorTrack(rawTracks.modelName,rawTracks.T.curves,initialPosition,"position");positionTrack!==void 0&&tracks.push(positionTrack)}if(rawTracks.R!==void 0&&Object.keys(rawTracks.R.curves).length>0){const rotationTrack=this.generateRotationTrack(rawTracks.modelName,rawTracks.R.curves,rawTracks.preRotation,rawTracks.postRotation,rawTracks.eulerOrder);rotationTrack!==void 0&&tracks.push(rotationTrack)}if(rawTracks.S!==void 0&&Object.keys(rawTracks.S.curves).length>0){const scaleTrack=this.generateVectorTrack(rawTracks.modelName,rawTracks.S.curves,initialScale,"scale");scaleTrack!==void 0&&tracks.push(scaleTrack)}if(rawTracks.DeformPercent!==void 0){const morphTrack=this.generateMorphTrack(rawTracks);morphTrack!==void 0&&tracks.push(morphTrack)}return tracks}generateVectorTrack(modelName,curves,initialValue,type){const times=this.getTimesForAllAxes(curves),values=this.getKeyframeTrackValues(times,curves,initialValue);return new VectorKeyframeTrack(modelName+"."+type,times,values)}generateRotationTrack(modelName,curves,preRotation,postRotation,eulerOrder){let times,values;if(curves.x!==void 0&&curves.y!==void 0&&curves.z!==void 0){const result=this.interpolateRotations(curves.x,curves.y,curves.z,eulerOrder);times=result[0],values=result[1]}const defaultEulerOrder=getEulerOrder(0);preRotation!==void 0&&(preRotation=preRotation.map(MathUtils.degToRad),preRotation.push(defaultEulerOrder),preRotation=new Euler().fromArray(preRotation),preRotation=new Quaternion().setFromEuler(preRotation)),postRotation!==void 0&&(postRotation=postRotation.map(MathUtils.degToRad),postRotation.push(defaultEulerOrder),postRotation=new Euler().fromArray(postRotation),postRotation=new Quaternion().setFromEuler(postRotation).invert());const quaternion=new Quaternion,euler=new Euler,quaternionValues=[];if(!values||!times)return new QuaternionKeyframeTrack(modelName+".quaternion",[0],[0]);for(let i=0;i<values.length;i+=3)euler.set(values[i],values[i+1],values[i+2],eulerOrder),quaternion.setFromEuler(euler),preRotation!==void 0&&quaternion.premultiply(preRotation),postRotation!==void 0&&quaternion.multiply(postRotation),i>2&&new Quaternion().fromArray(quaternionValues,(i-3)/3*4).dot(quaternion)<0&&quaternion.set(-quaternion.x,-quaternion.y,-quaternion.z,-quaternion.w),quaternion.toArray(quaternionValues,i/3*4);return new QuaternionKeyframeTrack(modelName+".quaternion",times,quaternionValues)}generateMorphTrack(rawTracks){const curves=rawTracks.DeformPercent.curves.morph,values=curves.values.map(function(val){return val/100}),morphNum=sceneGraph.getObjectByName(rawTracks.modelName).morphTargetDictionary[rawTracks.morphName];return new NumberKeyframeTrack(rawTracks.modelName+".morphTargetInfluences["+morphNum+"]",curves.times,values)}getTimesForAllAxes(curves){let times=[];if(curves.x!==void 0&&(times=times.concat(curves.x.times)),curves.y!==void 0&&(times=times.concat(curves.y.times)),curves.z!==void 0&&(times=times.concat(curves.z.times)),times=times.sort(function(a,b){return a-b}),times.length>1){let targetIndex=1,lastValue=times[0];for(let i=1;i<times.length;i++){const currentValue=times[i];currentValue!==lastValue&&(times[targetIndex]=currentValue,lastValue=currentValue,targetIndex++)}times=times.slice(0,targetIndex)}return times}getKeyframeTrackValues(times,curves,initialValue){const prevValue=initialValue,values=[];let xIndex=-1,yIndex=-1,zIndex=-1;return times.forEach(function(time){if(curves.x&&(xIndex=curves.x.times.indexOf(time)),curves.y&&(yIndex=curves.y.times.indexOf(time)),curves.z&&(zIndex=curves.z.times.indexOf(time)),xIndex!==-1){const xValue=curves.x.values[xIndex];values.push(xValue),prevValue[0]=xValue}else values.push(prevValue[0]);if(yIndex!==-1){const yValue=curves.y.values[yIndex];values.push(yValue),prevValue[1]=yValue}else values.push(prevValue[1]);if(zIndex!==-1){const zValue=curves.z.values[zIndex];values.push(zValue),prevValue[2]=zValue}else values.push(prevValue[2])}),values}interpolateRotations(curvex,curvey,curvez,eulerOrder){const times=[],values=[];times.push(curvex.times[0]),values.push(MathUtils.degToRad(curvex.values[0])),values.push(MathUtils.degToRad(curvey.values[0])),values.push(MathUtils.degToRad(curvez.values[0]));for(let i=1;i<curvex.values.length;i++){const initialValue=[curvex.values[i-1],curvey.values[i-1],curvez.values[i-1]];if(isNaN(initialValue[0])||isNaN(initialValue[1])||isNaN(initialValue[2]))continue;const initialValueRad=initialValue.map(MathUtils.degToRad),currentValue=[curvex.values[i],curvey.values[i],curvez.values[i]];if(isNaN(currentValue[0])||isNaN(currentValue[1])||isNaN(currentValue[2]))continue;const currentValueRad=currentValue.map(MathUtils.degToRad),valuesSpan=[currentValue[0]-initialValue[0],currentValue[1]-initialValue[1],currentValue[2]-initialValue[2]],absoluteSpan=[Math.abs(valuesSpan[0]),Math.abs(valuesSpan[1]),Math.abs(valuesSpan[2])];if(absoluteSpan[0]>=180||absoluteSpan[1]>=180||absoluteSpan[2]>=180){const numSubIntervals=Math.max(...absoluteSpan)/180,E1=new Euler(...initialValueRad,eulerOrder),E2=new Euler(...currentValueRad,eulerOrder),Q1=new Quaternion().setFromEuler(E1),Q2=new Quaternion().setFromEuler(E2);Q1.dot(Q2)&&Q2.set(-Q2.x,-Q2.y,-Q2.z,-Q2.w);const initialTime=curvex.times[i-1],timeSpan=curvex.times[i]-initialTime,Q=new Quaternion,E=new Euler;for(let t2=0;t2<1;t2+=1/numSubIntervals)Q.copy(Q1.clone().slerp(Q2.clone(),t2)),times.push(initialTime+t2*timeSpan),E.setFromQuaternion(Q,eulerOrder),values.push(E.x),values.push(E.y),values.push(E.z)}else times.push(curvex.times[i]),values.push(MathUtils.degToRad(curvex.values[i])),values.push(MathUtils.degToRad(curvey.values[i])),values.push(MathUtils.degToRad(curvez.values[i]))}return[times,values]}}class TextParser{static{__name(this,"TextParser")}getPrevNode(){return this.nodeStack[this.currentIndent-2]}getCurrentNode(){return this.nodeStack[this.currentIndent-1]}getCurrentProp(){return this.currentProp}pushStack(node){this.nodeStack.push(node),this.currentIndent+=1}popStack(){this.nodeStack.pop(),this.currentIndent-=1}setCurrentProp(val,name){this.currentProp=val,this.currentPropName=name}parse(text){this.currentIndent=0,this.allNodes=new FBXTree,this.nodeStack=[],this.currentProp=[],this.currentPropName="";const scope=this,split=text.split(/[\r\n]+/);return split.forEach(function(line,i){const matchComment=line.match(/^[\s\t]*;/),matchEmpty=line.match(/^[\s\t]*$/);if(matchComment||matchEmpty)return;const matchBeginning=line.match("^\\t{"+scope.currentIndent+"}(\\w+):(.*){",""),matchProperty=line.match("^\\t{"+scope.currentIndent+"}(\\w+):[\\s\\t\\r\\n](.*)"),matchEnd=line.match("^\\t{"+(scope.currentIndent-1)+"}}");matchBeginning?scope.parseNodeBegin(line,matchBeginning):matchProperty?scope.parseNodeProperty(line,matchProperty,split[++i]):matchEnd?scope.popStack():line.match(/^[^\s\t}]/)&&scope.parseNodePropertyContinued(line)}),this.allNodes}parseNodeBegin(line,property){const nodeName=property[1].trim().replace(/^"/,"").replace(/"$/,""),nodeAttrs=property[2].split(",").map(function(attr){return attr.trim().replace(/^"/,"").replace(/"$/,"")}),node={name:nodeName},attrs=this.parseNodeAttr(nodeAttrs),currentNode=this.getCurrentNode();this.currentIndent===0?this.allNodes.add(nodeName,node):nodeName in currentNode?(nodeName==="PoseNode"?currentNode.PoseNode.push(node):currentNode[nodeName].id!==void 0&&(currentNode[nodeName]={},currentNode[nodeName][currentNode[nodeName].id]=currentNode[nodeName]),attrs.id!==""&&(currentNode[nodeName][attrs.id]=node)):typeof attrs.id=="number"?(currentNode[nodeName]={},currentNode[nodeName][attrs.id]=node):nodeName!=="Properties70"&&(nodeName==="PoseNode"?currentNode[nodeName]=[node]:currentNode[nodeName]=node),typeof attrs.id=="number"&&(node.id=attrs.id),attrs.name!==""&&(node.attrName=attrs.name),attrs.type!==""&&(node.attrType=attrs.type),this.pushStack(node)}parseNodeAttr(attrs){let id2=attrs[0];attrs[0]!==""&&(id2=parseInt(attrs[0]),isNaN(id2)&&(id2=attrs[0]));let name="",type="";return attrs.length>1&&(name=attrs[1].replace(/^(\w+)::/,""),type=attrs[2]),{id:id2,name,type}}parseNodeProperty(line,property,contentLine){let propName=property[1].replace(/^"/,"").replace(/"$/,"").trim(),propValue=property[2].replace(/^"/,"").replace(/"$/,"").trim();propName==="Content"&&propValue===","&&(propValue=contentLine.replace(/"/g,"").replace(/,$/,"").trim());const currentNode=this.getCurrentNode();if(currentNode.name==="Properties70"){this.parseNodeSpecialProperty(line,propName,propValue);return}if(propName==="C"){const connProps=propValue.split(",").slice(1),from=parseInt(connProps[0]),to=parseInt(connProps[1]);let rest=propValue.split(",").slice(3);rest=rest.map(function(elem){return elem.trim().replace(/^"/,"")}),propName="connections",propValue=[from,to],append(propValue,rest),currentNode[propName]===void 0&&(currentNode[propName]=[])}propName==="Node"&&(currentNode.id=propValue),propName in currentNode&&Array.isArray(currentNode[propName])?currentNode[propName].push(propValue):propName!=="a"?currentNode[propName]=propValue:currentNode.a=propValue,this.setCurrentProp(currentNode,propName),propName==="a"&&propValue.slice(-1)!==","&&(currentNode.a=parseNumberArray(propValue))}parseNodePropertyContinued(line){const currentNode=this.getCurrentNode();currentNode.a+=line,line.slice(-1)!==","&&(currentNode.a=parseNumberArray(currentNode.a))}parseNodeSpecialProperty(line,propName,propValue){const props=propValue.split('",').map(function(prop){return prop.trim().replace(/^\"/,"").replace(/\s/,"_")}),innerPropName=props[0],innerPropType1=props[1],innerPropType2=props[2],innerPropFlag=props[3];let innerPropValue=props[4];switch(innerPropType1){case"int":case"enum":case"bool":case"ULongLong":case"double":case"Number":case"FieldOfView":innerPropValue=parseFloat(innerPropValue);break;case"Color":case"ColorRGB":case"Vector3D":case"Lcl_Translation":case"Lcl_Rotation":case"Lcl_Scaling":innerPropValue=parseNumberArray(innerPropValue);break}this.getPrevNode()[innerPropName]={type:innerPropType1,type2:innerPropType2,flag:innerPropFlag,value:innerPropValue},this.setCurrentProp(this.getPrevNode(),innerPropName)}}class BinaryParser{static{__name(this,"BinaryParser")}parse(buffer){const reader=new BinaryReader(buffer);reader.skip(23);const version=reader.getUint32();if(version<6400)throw new Error("THREE.FBXLoader: FBX version not supported, FileVersion: "+version);const allNodes=new FBXTree;for(;!this.endOfContent(reader);){const node=this.parseNode(reader,version);node!==null&&allNodes.add(node.name,node)}return allNodes}endOfContent(reader){return reader.size()%16===0?(reader.getOffset()+160+16&-16)>=reader.size():reader.getOffset()+160+16>=reader.size()}parseNode(reader,version){const node={},endOffset=version>=7500?reader.getUint64():reader.getUint32(),numProperties=version>=7500?reader.getUint64():reader.getUint32();version>=7500?reader.getUint64():reader.getUint32();const nameLen=reader.getUint8(),name=reader.getString(nameLen);if(endOffset===0)return null;const propertyList=[];for(let i=0;i<numProperties;i++)propertyList.push(this.parseProperty(reader));const id2=propertyList.length>0?propertyList[0]:"",attrName=propertyList.length>1?propertyList[1]:"",attrType=propertyList.length>2?propertyList[2]:"";for(node.singleProperty=numProperties===1&&reader.getOffset()===endOffset;endOffset>reader.getOffset();){const subNode=this.parseNode(reader,version);subNode!==null&&this.parseSubNode(name,node,subNode)}return node.propertyList=propertyList,typeof id2=="number"&&(node.id=id2),attrName!==""&&(node.attrName=attrName),attrType!==""&&(node.attrType=attrType),name!==""&&(node.name=name),node}parseSubNode(name,node,subNode){if(subNode.singleProperty===!0){const value=subNode.propertyList[0];Array.isArray(value)?(node[subNode.name]=subNode,subNode.a=value):node[subNode.name]=value}else if(name==="Connections"&&subNode.name==="C"){const array=[];subNode.propertyList.forEach(function(property,i){i!==0&&array.push(property)}),node.connections===void 0&&(node.connections=[]),node.connections.push(array)}else if(subNode.name==="Properties70")Object.keys(subNode).forEach(function(key){node[key]=subNode[key]});else if(name==="Properties70"&&subNode.name==="P"){let innerPropName=subNode.propertyList[0],innerPropType1=subNode.propertyList[1];const innerPropType2=subNode.propertyList[2],innerPropFlag=subNode.propertyList[3];let innerPropValue;innerPropName.indexOf("Lcl ")===0&&(innerPropName=innerPropName.replace("Lcl ","Lcl_")),innerPropType1.indexOf("Lcl ")===0&&(innerPropType1=innerPropType1.replace("Lcl ","Lcl_")),innerPropType1==="Color"||innerPropType1==="ColorRGB"||innerPropType1==="Vector"||innerPropType1==="Vector3D"||innerPropType1.indexOf("Lcl_")===0?innerPropValue=[subNode.propertyList[4],subNode.propertyList[5],subNode.propertyList[6]]:innerPropValue=subNode.propertyList[4],node[innerPropName]={type:innerPropType1,type2:innerPropType2,flag:innerPropFlag,value:innerPropValue}}else node[subNode.name]===void 0?typeof subNode.id=="number"?(node[subNode.name]={},node[subNode.name][subNode.id]=subNode):node[subNode.name]=subNode:subNode.name==="PoseNode"?(Array.isArray(node[subNode.name])||(node[subNode.name]=[node[subNode.name]]),node[subNode.name].push(subNode)):node[subNode.name][subNode.id]===void 0&&(node[subNode.name][subNode.id]=subNode)}parseProperty(reader){const type=reader.getString(1);let length;switch(type){case"C":return reader.getBoolean();case"D":return reader.getFloat64();case"F":return reader.getFloat32();case"I":return reader.getInt32();case"L":return reader.getInt64();case"R":return length=reader.getUint32(),reader.getArrayBuffer(length);case"S":return length=reader.getUint32(),reader.getString(length);case"Y":return reader.getInt16();case"b":case"c":case"d":case"f":case"i":case"l":const arrayLength=reader.getUint32(),encoding=reader.getUint32(),compressedLength=reader.getUint32();if(encoding===0)switch(type){case"b":case"c":return reader.getBooleanArray(arrayLength);case"d":return reader.getFloat64Array(arrayLength);case"f":return reader.getFloat32Array(arrayLength);case"i":return reader.getInt32Array(arrayLength);case"l":return reader.getInt64Array(arrayLength)}const data=unzlibSync(new Uint8Array(reader.getArrayBuffer(compressedLength))),reader2=new BinaryReader(data.buffer);switch(type){case"b":case"c":return reader2.getBooleanArray(arrayLength);case"d":return reader2.getFloat64Array(arrayLength);case"f":return reader2.getFloat32Array(arrayLength);case"i":return reader2.getInt32Array(arrayLength);case"l":return reader2.getInt64Array(arrayLength)}break;default:throw new Error("THREE.FBXLoader: Unknown property type "+type)}}}class BinaryReader{static{__name(this,"BinaryReader")}constructor(buffer,littleEndian){this.dv=new DataView(buffer),this.offset=0,this.littleEndian=littleEndian!==void 0?littleEndian:!0,this._textDecoder=new TextDecoder}getOffset(){return this.offset}size(){return this.dv.buffer.byteLength}skip(length){this.offset+=length}getBoolean(){return(this.getUint8()&1)===1}getBooleanArray(size){const a=[];for(let i=0;i<size;i++)a.push(this.getBoolean());return a}getUint8(){const value=this.dv.getUint8(this.offset);return this.offset+=1,value}getInt16(){const value=this.dv.getInt16(this.offset,this.littleEndian);return this.offset+=2,value}getInt32(){const value=this.dv.getInt32(this.offset,this.littleEndian);return this.offset+=4,value}getInt32Array(size){const a=[];for(let i=0;i<size;i++)a.push(this.getInt32());return a}getUint32(){const value=this.dv.getUint32(this.offset,this.littleEndian);return this.offset+=4,value}getInt64(){let low,high;return this.littleEndian?(low=this.getUint32(),high=this.getUint32()):(high=this.getUint32(),low=this.getUint32()),high&2147483648?(high=~high&4294967295,low=~low&4294967295,low===4294967295&&(high=high+1&4294967295),low=low+1&4294967295,-(high*4294967296+low)):high*4294967296+low}getInt64Array(size){const a=[];for(let i=0;i<size;i++)a.push(this.getInt64());return a}getUint64(){let low,high;return this.littleEndian?(low=this.getUint32(),high=this.getUint32()):(high=this.getUint32(),low=this.getUint32()),high*4294967296+low}getFloat32(){const value=this.dv.getFloat32(this.offset,this.littleEndian);return this.offset+=4,value}getFloat32Array(size){const a=[];for(let i=0;i<size;i++)a.push(this.getFloat32());return a}getFloat64(){const value=this.dv.getFloat64(this.offset,this.littleEndian);return this.offset+=8,value}getFloat64Array(size){const a=[];for(let i=0;i<size;i++)a.push(this.getFloat64());return a}getArrayBuffer(size){const value=this.dv.buffer.slice(this.offset,this.offset+size);return this.offset+=size,value}getString(size){const start=this.offset;let a=new Uint8Array(this.dv.buffer,start,size);this.skip(size);const nullByte=a.indexOf(0);return nullByte>=0&&(a=new Uint8Array(this.dv.buffer,start,nullByte)),this._textDecoder.decode(a)}}class FBXTree{static{__name(this,"FBXTree")}add(key,val){this[key]=val}}function isFbxFormatBinary(buffer){const CORRECT="Kaydara FBX Binary  \0";return buffer.byteLength>=CORRECT.length&&CORRECT===convertArrayBufferToString(buffer,0,CORRECT.length)}__name(isFbxFormatBinary,"isFbxFormatBinary");function isFbxFormatASCII(text){const CORRECT=["K","a","y","d","a","r","a","\\","F","B","X","\\","B","i","n","a","r","y","\\","\\"];let cursor=0;function read(offset){const result=text[offset-1];return text=text.slice(cursor+offset),cursor++,result}__name(read,"read");for(let i=0;i<CORRECT.length;++i)if(read(1)===CORRECT[i])return!1;return!0}__name(isFbxFormatASCII,"isFbxFormatASCII");function getFbxVersion(text){const versionRegExp=/FBXVersion: (\d+)/,match=text.match(versionRegExp);if(match)return parseInt(match[1]);throw new Error("THREE.FBXLoader: Cannot find the version number for the file given.")}__name(getFbxVersion,"getFbxVersion");function convertFBXTimeToSeconds(time){return time/46186158e3}__name(convertFBXTimeToSeconds,"convertFBXTimeToSeconds");const dataArray=[];function getData(polygonVertexIndex,polygonIndex,vertexIndex,infoObject){let index;switch(infoObject.mappingType){case"ByPolygonVertex":index=polygonVertexIndex;break;case"ByPolygon":index=polygonIndex;break;case"ByVertice":index=vertexIndex;break;case"AllSame":index=infoObject.indices[0];break;default:console.warn("THREE.FBXLoader: unknown attribute mapping type "+infoObject.mappingType)}infoObject.referenceType==="IndexToDirect"&&(index=infoObject.indices[index]);const from=index*infoObject.dataSize,to=from+infoObject.dataSize;return slice(dataArray,infoObject.buffer,from,to)}__name(getData,"getData");const tempEuler=new Euler,tempVec=new Vector3;function generateTransform(transformData){const lTranslationM=new Matrix4,lPreRotationM=new Matrix4,lRotationM=new Matrix4,lPostRotationM=new Matrix4,lScalingM=new Matrix4,lScalingPivotM=new Matrix4,lScalingOffsetM=new Matrix4,lRotationOffsetM=new Matrix4,lRotationPivotM=new Matrix4,lParentGX=new Matrix4,lParentLX=new Matrix4,lGlobalT=new Matrix4,inheritType=transformData.inheritType?transformData.inheritType:0;transformData.translation&&lTranslationM.setPosition(tempVec.fromArray(transformData.translation));const defaultEulerOrder=getEulerOrder(0);if(transformData.preRotation){const array=transformData.preRotation.map(MathUtils.degToRad);array.push(defaultEulerOrder),lPreRotationM.makeRotationFromEuler(tempEuler.fromArray(array))}if(transformData.rotation){const array=transformData.rotation.map(MathUtils.degToRad);array.push(transformData.eulerOrder||defaultEulerOrder),lRotationM.makeRotationFromEuler(tempEuler.fromArray(array))}if(transformData.postRotation){const array=transformData.postRotation.map(MathUtils.degToRad);array.push(defaultEulerOrder),lPostRotationM.makeRotationFromEuler(tempEuler.fromArray(array)),lPostRotationM.invert()}transformData.scale&&lScalingM.scale(tempVec.fromArray(transformData.scale)),transformData.scalingOffset&&lScalingOffsetM.setPosition(tempVec.fromArray(transformData.scalingOffset)),transformData.scalingPivot&&lScalingPivotM.setPosition(tempVec.fromArray(transformData.scalingPivot)),transformData.rotationOffset&&lRotationOffsetM.setPosition(tempVec.fromArray(transformData.rotationOffset)),transformData.rotationPivot&&lRotationPivotM.setPosition(tempVec.fromArray(transformData.rotationPivot)),transformData.parentMatrixWorld&&(lParentLX.copy(transformData.parentMatrix),lParentGX.copy(transformData.parentMatrixWorld));const lLRM=lPreRotationM.clone().multiply(lRotationM).multiply(lPostRotationM),lParentGRM=new Matrix4;lParentGRM.extractRotation(lParentGX);const lParentTM=new Matrix4;lParentTM.copyPosition(lParentGX);const lParentGRSM=lParentTM.clone().invert().multiply(lParentGX),lParentGSM=lParentGRM.clone().invert().multiply(lParentGRSM),lLSM=lScalingM,lGlobalRS=new Matrix4;if(inheritType===0)lGlobalRS.copy(lParentGRM).multiply(lLRM).multiply(lParentGSM).multiply(lLSM);else if(inheritType===1)lGlobalRS.copy(lParentGRM).multiply(lParentGSM).multiply(lLRM).multiply(lLSM);else{const lParentLSM_inv=new Matrix4().scale(new Vector3().setFromMatrixScale(lParentLX)).clone().invert(),lParentGSM_noLocal=lParentGSM.clone().multiply(lParentLSM_inv);lGlobalRS.copy(lParentGRM).multiply(lLRM).multiply(lParentGSM_noLocal).multiply(lLSM)}const lRotationPivotM_inv=lRotationPivotM.clone().invert(),lScalingPivotM_inv=lScalingPivotM.clone().invert();let lTransform=lTranslationM.clone().multiply(lRotationOffsetM).multiply(lRotationPivotM).multiply(lPreRotationM).multiply(lRotationM).multiply(lPostRotationM).multiply(lRotationPivotM_inv).multiply(lScalingOffsetM).multiply(lScalingPivotM).multiply(lScalingM).multiply(lScalingPivotM_inv);const lLocalTWithAllPivotAndOffsetInfo=new Matrix4().copyPosition(lTransform),lGlobalTranslation=lParentGX.clone().multiply(lLocalTWithAllPivotAndOffsetInfo);return lGlobalT.copyPosition(lGlobalTranslation),lTransform=lGlobalT.clone().multiply(lGlobalRS),lTransform.premultiply(lParentGX.invert()),lTransform}__name(generateTransform,"generateTransform");function getEulerOrder(order){order=order||0;const enums=["ZYX","YZX","XZY","ZXY","YXZ","XYZ"];return order===6?(console.warn("THREE.FBXLoader: unsupported Euler Order: Spherical XYZ. Animations and rotations may be incorrect."),enums[0]):enums[order]}__name(getEulerOrder,"getEulerOrder");function parseNumberArray(value){return value.split(",").map(function(val){return parseFloat(val)})}__name(parseNumberArray,"parseNumberArray");function convertArrayBufferToString(buffer,from,to){return from===void 0&&(from=0),to===void 0&&(to=buffer.byteLength),new TextDecoder().decode(new Uint8Array(buffer,from,to))}__name(convertArrayBufferToString,"convertArrayBufferToString");function append(a,b){for(let i=0,j=a.length,l=b.length;i<l;i++,j++)a[j]=b[i]}__name(append,"append");function slice(a,b,from,to){for(let i=from,j=0;i<to;i++,j++)a[j]=b[i];return a}__name(slice,"slice");class STLLoader extends Loader{static{__name(this,"STLLoader")}constructor(manager){super(manager)}load(url,onLoad,onProgress,onError){const scope=this,loader=new FileLoader(this.manager);loader.setPath(this.path),loader.setResponseType("arraybuffer"),loader.setRequestHeader(this.requestHeader),loader.setWithCredentials(this.withCredentials),loader.load(url,function(text){try{onLoad(scope.parse(text))}catch(e){onError?onError(e):console.error(e),scope.manager.itemError(url)}},onProgress,onError)}parse(data){function isBinary(data2){const reader=new DataView(data2),face_size=32/8*3+32/8*3*3+16/8,n_faces=reader.getUint32(80,!0);if(80+32/8+n_faces*face_size===reader.byteLength)return!0;const solid=[115,111,108,105,100];for(let off=0;off<5;off++)if(matchDataViewAt(solid,reader,off))return!1;return!0}__name(isBinary,"isBinary");function matchDataViewAt(query,reader,offset){for(let i=0,il=query.length;i<il;i++)if(query[i]!==reader.getUint8(offset+i))return!1;return!0}__name(matchDataViewAt,"matchDataViewAt");function parseBinary(data2){const reader=new DataView(data2),faces=reader.getUint32(80,!0);let r,g,b,hasColors=!1,colors,defaultR,defaultG,defaultB,alpha;for(let index=0;index<70;index++)reader.getUint32(index,!1)==1129270351&&reader.getUint8(index+4)==82&&reader.getUint8(index+5)==61&&(hasColors=!0,colors=new Float32Array(faces*3*3),defaultR=reader.getUint8(index+6)/255,defaultG=reader.getUint8(index+7)/255,defaultB=reader.getUint8(index+8)/255,alpha=reader.getUint8(index+9)/255);const dataOffset=84,faceLength=12*4+2,geometry=new BufferGeometry,vertices=new Float32Array(faces*3*3),normals=new Float32Array(faces*3*3),color=new Color;for(let face=0;face<faces;face++){const start=dataOffset+face*faceLength,normalX=reader.getFloat32(start,!0),normalY=reader.getFloat32(start+4,!0),normalZ=reader.getFloat32(start+8,!0);if(hasColors){const packedColor=reader.getUint16(start+48,!0);packedColor&32768?(r=defaultR,g=defaultG,b=defaultB):(r=(packedColor&31)/31,g=(packedColor>>5&31)/31,b=(packedColor>>10&31)/31)}for(let i=1;i<=3;i++){const vertexstart=start+i*12,componentIdx=face*3*3+(i-1)*3;vertices[componentIdx]=reader.getFloat32(vertexstart,!0),vertices[componentIdx+1]=reader.getFloat32(vertexstart+4,!0),vertices[componentIdx+2]=reader.getFloat32(vertexstart+8,!0),normals[componentIdx]=normalX,normals[componentIdx+1]=normalY,normals[componentIdx+2]=normalZ,hasColors&&(color.setRGB(r,g,b,SRGBColorSpace),colors[componentIdx]=color.r,colors[componentIdx+1]=color.g,colors[componentIdx+2]=color.b)}}return geometry.setAttribute("position",new BufferAttribute(vertices,3)),geometry.setAttribute("normal",new BufferAttribute(normals,3)),hasColors&&(geometry.setAttribute("color",new BufferAttribute(colors,3)),geometry.hasColors=!0,geometry.alpha=alpha),geometry}__name(parseBinary,"parseBinary");function parseASCII(data2){const geometry=new BufferGeometry,patternSolid=/solid([\s\S]*?)endsolid/g,patternFace=/facet([\s\S]*?)endfacet/g,patternName=/solid\s(.+)/;let faceCounter=0;const patternFloat=/[\s]+([+-]?(?:\d*)(?:\.\d*)?(?:[eE][+-]?\d+)?)/.source,patternVertex=new RegExp("vertex"+patternFloat+patternFloat+patternFloat,"g"),patternNormal=new RegExp("normal"+patternFloat+patternFloat+patternFloat,"g"),vertices=[],normals=[],groupNames=[],normal=new Vector3;let result,groupCount=0,startVertex=0,endVertex=0;for(;(result=patternSolid.exec(data2))!==null;){startVertex=endVertex;const solid=result[0],name=(result=patternName.exec(solid))!==null?result[1]:"";for(groupNames.push(name);(result=patternFace.exec(solid))!==null;){let vertexCountPerFace=0,normalCountPerFace=0;const text=result[0];for(;(result=patternNormal.exec(text))!==null;)normal.x=parseFloat(result[1]),normal.y=parseFloat(result[2]),normal.z=parseFloat(result[3]),normalCountPerFace++;for(;(result=patternVertex.exec(text))!==null;)vertices.push(parseFloat(result[1]),parseFloat(result[2]),parseFloat(result[3])),normals.push(normal.x,normal.y,normal.z),vertexCountPerFace++,endVertex++;normalCountPerFace!==1&&console.error("THREE.STLLoader: Something isn't right with the normal of face number "+faceCounter),vertexCountPerFace!==3&&console.error("THREE.STLLoader: Something isn't right with the vertices of face number "+faceCounter),faceCounter++}const start=startVertex,count=endVertex-startVertex;geometry.userData.groupNames=groupNames,geometry.addGroup(start,count,groupCount),groupCount++}return geometry.setAttribute("position",new Float32BufferAttribute(vertices,3)),geometry.setAttribute("normal",new Float32BufferAttribute(normals,3)),geometry}__name(parseASCII,"parseASCII");function ensureString(buffer){return typeof buffer!="string"?new TextDecoder().decode(buffer):buffer}__name(ensureString,"ensureString");function ensureBinary(buffer){if(typeof buffer=="string"){const array_buffer=new Uint8Array(buffer.length);for(let i=0;i<buffer.length;i++)array_buffer[i]=buffer.charCodeAt(i)&255;return array_buffer.buffer||array_buffer}else return buffer}__name(ensureBinary,"ensureBinary");const binData=ensureBinary(data);return isBinary(binData)?parseBinary(binData):parseASCII(ensureString(data))}}async function uploadFile(load3d,file2,fileInput){let uploadPath;try{const body=new FormData;body.append("image",file2),body.append("subfolder","3d");const resp=await api.fetchApi("/upload/image",{method:"POST",body});if(resp.status===200){const data=await resp.json();let path=data.name;data.subfolder&&(path=data.subfolder+"/"+path),uploadPath=path;const modelUrl=api.apiURL(getResourceURL(...splitFilePath(path),"input"));if(await load3d.loadModel(modelUrl,file2.name),file2.name.split(".").pop()?.toLowerCase()==="obj"&&fileInput?.files)try{const mtlFile=Array.from(fileInput.files).find(f=>f.name.toLowerCase().endsWith(".mtl"));if(mtlFile){const mtlFormData=new FormData;mtlFormData.append("image",mtlFile),mtlFormData.append("subfolder","3d"),await api.fetchApi("/upload/image",{method:"POST",body:mtlFormData})}}catch(mtlError){console.warn("Failed to upload MTL file:",mtlError)}}else useToastStore().addAlert(resp.status+" - "+resp.statusText)}catch(error){console.error("Upload error:",error),useToastStore().addAlert(error instanceof Error?error.message:"Upload failed")}return uploadPath}__name(uploadFile,"uploadFile");class Load3d{static{__name(this,"Load3d")}scene;perspectiveCamera;orthographicCamera;activeCamera;renderer;controls;gltfLoader;objLoader;mtlLoader;fbxLoader;stlLoader;currentModel=null;originalModel=null;node;animationFrameId=null;gridHelper;lights=[];clock;normalMaterial;standardMaterial;wireframeMaterial;depthMaterial;originalMaterials=new WeakMap;materialMode="original";currentUpDirection="original";originalRotation=null;constructor(container){this.scene=new Scene,this.perspectiveCamera=new PerspectiveCamera(75,1,.1,1e3),this.perspectiveCamera.position.set(5,5,5);const frustumSize=10;this.orthographicCamera=new OrthographicCamera(-frustumSize/2,frustumSize/2,frustumSize/2,-frustumSize/2,.1,1e3),this.orthographicCamera.position.set(5,5,5),this.activeCamera=this.perspectiveCamera,this.perspectiveCamera.lookAt(0,0,0),this.orthographicCamera.lookAt(0,0,0),this.renderer=new WebGLRenderer({antialias:!0}),this.renderer.setSize(300,300),this.renderer.setClearColor(2631720);const rendererDomElement=this.renderer.domElement;container.appendChild(rendererDomElement),this.controls=new OrbitControls(this.activeCamera,this.renderer.domElement),this.controls.enableDamping=!0,this.gltfLoader=new GLTFLoader,this.objLoader=new OBJLoader,this.mtlLoader=new MTLLoader,this.fbxLoader=new FBXLoader,this.stlLoader=new STLLoader,this.clock=new Clock,this.setupLights(),this.gridHelper=new GridHelper(10,10),this.gridHelper.position.set(0,0,0),this.scene.add(this.gridHelper),this.normalMaterial=new MeshNormalMaterial({flatShading:!1,side:DoubleSide,normalScale:new Vector2(1,1),transparent:!1,opacity:1}),this.wireframeMaterial=new MeshBasicMaterial({color:16777215,wireframe:!0,transparent:!1,opacity:1}),this.depthMaterial=new MeshDepthMaterial({depthPacking:BasicDepthPacking,side:DoubleSide}),this.standardMaterial=this.createSTLMaterial(),this.animate(),this.handleResize(),this.startAnimation()}getCameraState(){const currentType=this.getCurrentCameraType();return{position:this.activeCamera.position.clone(),target:this.controls.target.clone(),zoom:this.activeCamera instanceof OrthographicCamera?this.activeCamera.zoom:this.activeCamera.zoom,cameraType:currentType}}setCameraState(state){this.activeCamera!==(state.cameraType==="perspective"?this.perspectiveCamera:this.orthographicCamera)&&this.toggleCamera(state.cameraType),this.activeCamera.position.copy(state.position),this.controls.target.copy(state.target),this.activeCamera instanceof OrthographicCamera?(this.activeCamera.zoom=state.zoom,this.activeCamera.updateProjectionMatrix()):this.activeCamera instanceof PerspectiveCamera&&(this.activeCamera.zoom=state.zoom,this.activeCamera.updateProjectionMatrix()),this.controls.update()}setUpDirection(direction){if(this.currentModel){switch(!this.originalRotation&&this.currentModel.rotation&&(this.originalRotation=this.currentModel.rotation.clone()),this.currentUpDirection=direction,this.originalRotation&&this.currentModel.rotation.copy(this.originalRotation),direction){case"original":break;case"-x":this.currentModel.rotation.z=Math.PI/2;break;case"+x":this.currentModel.rotation.z=-Math.PI/2;break;case"-y":this.currentModel.rotation.x=Math.PI;break;case"+y":break;case"-z":this.currentModel.rotation.x=Math.PI/2;break;case"+z":this.currentModel.rotation.x=-Math.PI/2;break}this.renderer.render(this.scene,this.activeCamera)}}setMaterialMode(mode){this.materialMode=mode,this.currentModel&&(mode==="depth"?this.renderer.outputColorSpace=LinearSRGBColorSpace:this.renderer.outputColorSpace=SRGBColorSpace,this.currentModel.traverse(child=>{if(child instanceof Mesh)switch(mode){case"depth":this.originalMaterials.has(child)||this.originalMaterials.set(child,child.material);const depthMat=new MeshDepthMaterial({depthPacking:BasicDepthPacking,side:DoubleSide});depthMat.onBeforeCompile=shader=>{shader.uniforms.cameraType={value:this.activeCamera instanceof OrthographicCamera?1:0},shader.fragmentShader=`
                  uniform float cameraType;
                  ${shader.fragmentShader}
                `,shader.fragmentShader=shader.fragmentShader.replace(/gl_FragColor\s*=\s*vec4\(\s*vec3\(\s*1.0\s*-\s*fragCoordZ\s*\)\s*,\s*opacity\s*\)\s*;/,`
                    float depth = 1.0 - fragCoordZ;
                    if (cameraType > 0.5) {
                      depth = pow(depth, 400.0);
                    } else {
                      depth = pow(depth, 0.6);
                    }
                    gl_FragColor = vec4(vec3(depth), opacity);
                  `)},depthMat.customProgramCacheKey=()=>this.activeCamera instanceof OrthographicCamera?"ortho":"persp",child.material=depthMat;break;case"normal":this.originalMaterials.has(child)||this.originalMaterials.set(child,child.material),child.material=new MeshNormalMaterial({flatShading:!1,side:DoubleSide,normalScale:new Vector2(1,1),transparent:!1,opacity:1}),child.geometry.computeVertexNormals();break;case"wireframe":this.originalMaterials.has(child)||this.originalMaterials.set(child,child.material),child.material=new MeshBasicMaterial({color:16777215,wireframe:!0,transparent:!1,opacity:1});break;case"original":const originalMaterial=this.originalMaterials.get(child);originalMaterial?child.material=originalMaterial:child.material=this.standardMaterial;break}}),this.renderer.render(this.scene,this.activeCamera))}setupLights(){const ambientLight=new AmbientLight(16777215,.5);this.scene.add(ambientLight),this.lights.push(ambientLight);const mainLight=new DirectionalLight(16777215,.8);mainLight.position.set(0,10,10),this.scene.add(mainLight),this.lights.push(mainLight);const backLight=new DirectionalLight(16777215,.5);backLight.position.set(0,10,-10),this.scene.add(backLight),this.lights.push(backLight);const leftFillLight=new DirectionalLight(16777215,.3);leftFillLight.position.set(-10,0,0),this.scene.add(leftFillLight),this.lights.push(leftFillLight);const rightFillLight=new DirectionalLight(16777215,.3);rightFillLight.position.set(10,0,0),this.scene.add(rightFillLight),this.lights.push(rightFillLight);const bottomLight=new DirectionalLight(16777215,.2);bottomLight.position.set(0,-10,0),this.scene.add(bottomLight),this.lights.push(bottomLight)}toggleCamera(cameraType){const oldCamera=this.activeCamera,position=oldCamera.position.clone(),rotation=oldCamera.rotation.clone(),target=this.controls.target.clone();if(!cameraType)this.activeCamera=oldCamera===this.perspectiveCamera?this.orthographicCamera:this.perspectiveCamera;else if(this.activeCamera=cameraType==="perspective"?this.perspectiveCamera:this.orthographicCamera,oldCamera===this.activeCamera)return;this.activeCamera.position.copy(position),this.activeCamera.rotation.copy(rotation),this.materialMode==="depth"&&oldCamera!==this.activeCamera&&this.setMaterialMode("depth"),this.controls.object=this.activeCamera,this.controls.target.copy(target),this.controls.update(),this.handleResize()}getCurrentCameraType(){return this.activeCamera===this.perspectiveCamera?"perspective":"orthographic"}toggleGrid(showGrid){this.gridHelper&&(this.gridHelper.visible=showGrid)}setLightIntensity(intensity){this.lights.forEach(light=>{light instanceof DirectionalLight?light===this.lights[1]?light.intensity=intensity*.8:light===this.lights[2]?light.intensity=intensity*.5:light===this.lights[5]?light.intensity=intensity*.2:light.intensity=intensity*.3:light instanceof AmbientLight&&(light.intensity=intensity*.5)})}startAnimation(){const animate=__name(()=>{this.animationFrameId=requestAnimationFrame(animate),this.controls.update(),this.renderer.render(this.scene,this.activeCamera)},"animate");animate()}clearModel(){const objectsToRemove=[];this.scene.traverse(object=>{object===this.gridHelper||this.lights.includes(object)||object===this.perspectiveCamera||object===this.orthographicCamera||objectsToRemove.push(object)}),objectsToRemove.forEach(obj=>{obj.parent&&obj.parent!==this.scene?obj.parent.remove(obj):this.scene.remove(obj),obj instanceof Mesh&&(obj.geometry?.dispose(),Array.isArray(obj.material)?obj.material.forEach(material=>material.dispose()):obj.material?.dispose())}),this.resetScene()}resetScene(){this.currentModel=null,this.originalRotation=null;const defaultDistance=10;this.perspectiveCamera.position.set(defaultDistance,defaultDistance,defaultDistance),this.orthographicCamera.position.set(defaultDistance,defaultDistance,defaultDistance),this.perspectiveCamera.lookAt(0,0,0),this.orthographicCamera.lookAt(0,0,0);const frustumSize=10,aspect2=this.renderer.domElement.width/this.renderer.domElement.height;this.orthographicCamera.left=-frustumSize*aspect2/2,this.orthographicCamera.right=frustumSize*aspect2/2,this.orthographicCamera.top=frustumSize/2,this.orthographicCamera.bottom=-frustumSize/2,this.perspectiveCamera.updateProjectionMatrix(),this.orthographicCamera.updateProjectionMatrix(),this.controls.target.set(0,0,0),this.controls.update(),this.renderer.render(this.scene,this.activeCamera),this.materialMode="original",this.originalMaterials=new WeakMap,this.renderer.outputColorSpace=SRGBColorSpace}remove(){this.animationFrameId!==null&&cancelAnimationFrame(this.animationFrameId),this.controls.dispose(),this.renderer.dispose(),this.renderer.domElement.remove(),this.scene.clear()}async loadModelInternal(url,fileExtension){let model=null;switch(fileExtension){case"stl":const geometry=await this.stlLoader.loadAsync(url);this.originalModel=geometry,geometry.computeVertexNormals();const mesh=new Mesh(geometry,this.standardMaterial),group=new Group;group.add(mesh),model=group;break;case"fbx":const fbxModel=await this.fbxLoader.loadAsync(url);this.originalModel=fbxModel,model=fbxModel,fbxModel.traverse(child=>{child instanceof Mesh&&this.originalMaterials.set(child,child.material)});break;case"obj":if(this.materialMode==="original"){const mtlUrl=url.replace(/\.obj([^.]*$)/,".mtl$1");try{const materials=await this.mtlLoader.loadAsync(mtlUrl);materials.preload(),this.objLoader.setMaterials(materials)}catch{console.log("No MTL file found or error loading it, continuing without materials")}}model=await this.objLoader.loadAsync(url),model.traverse(child=>{child instanceof Mesh&&this.originalMaterials.set(child,child.material)});break;case"gltf":case"glb":const gltf=await this.gltfLoader.loadAsync(url);this.originalModel=gltf,model=gltf.scene,gltf.scene.traverse(child=>{child instanceof Mesh&&(child.geometry.computeVertexNormals(),this.originalMaterials.set(child,child.material))});break}return model}async loadModel(url,originalFileName){try{this.clearModel();let fileExtension;if(originalFileName?fileExtension=originalFileName.split(".").pop()?.toLowerCase():fileExtension=new URLSearchParams(url.split("?")[1]).get("filename")?.split(".").pop()?.toLowerCase(),!fileExtension){useToastStore().addAlert("Could not determine file type");return}let model=await this.loadModelInternal(url,fileExtension);model&&(this.currentModel=model,await this.setupModel(model))}catch(error){console.error("Error loading model:",error)}}async setupModel(model){const box=new Box3().setFromObject(model),size=box.getSize(new Vector3),center=box.getCenter(new Vector3),scale=5/Math.max(size.x,size.y,size.z);model.scale.multiplyScalar(scale),box.setFromObject(model),box.getCenter(center),box.getSize(size),model.position.set(-center.x,-box.min.y,-center.z),this.scene.add(model),this.materialMode!=="original"&&this.setMaterialMode(this.materialMode),this.currentUpDirection!=="original"&&this.setUpDirection(this.currentUpDirection),await this.setupCamera(size)}async setupCamera(size){const distance=Math.max(size.x,size.z)*2,height=size.y*2;if(this.perspectiveCamera.position.set(distance,height,distance),this.orthographicCamera.position.set(distance,height,distance),this.activeCamera===this.perspectiveCamera)this.perspectiveCamera.lookAt(0,size.y/2,0),this.perspectiveCamera.updateProjectionMatrix();else{const frustumSize=Math.max(size.x,size.y,size.z)*2,aspect2=this.renderer.domElement.width/this.renderer.domElement.height;this.orthographicCamera.left=-frustumSize*aspect2/2,this.orthographicCamera.right=frustumSize*aspect2/2,this.orthographicCamera.top=frustumSize/2,this.orthographicCamera.bottom=-frustumSize/2,this.orthographicCamera.lookAt(0,size.y/2,0),this.orthographicCamera.updateProjectionMatrix()}this.controls.target.set(0,size.y/2,0),this.controls.update(),this.renderer.outputColorSpace=SRGBColorSpace,this.renderer.toneMapping=ACESFilmicToneMapping,this.renderer.toneMappingExposure=1,this.handleResize()}handleResize(){const parentElement=this.renderer?.domElement?.parentElement;if(!parentElement){console.warn("Parent element not found");return}const width=parentElement?.clientWidth,height=parentElement?.clientHeight;if(this.activeCamera===this.perspectiveCamera)this.perspectiveCamera.aspect=width/height,this.perspectiveCamera.updateProjectionMatrix();else{const aspect2=width/height;this.orthographicCamera.left=-10*aspect2/2,this.orthographicCamera.right=10*aspect2/2,this.orthographicCamera.top=10/2,this.orthographicCamera.bottom=-10/2,this.orthographicCamera.updateProjectionMatrix()}this.renderer.setSize(width,height)}animate=__name(()=>{requestAnimationFrame(this.animate),this.controls.update(),this.renderer.render(this.scene,this.activeCamera)},"animate");captureScene(width,height){return new Promise((resolve,reject)=>{try{const originalWidth=this.renderer.domElement.width,originalHeight=this.renderer.domElement.height;if(this.renderer.setSize(width,height),this.activeCamera===this.perspectiveCamera)this.perspectiveCamera.aspect=width/height,this.perspectiveCamera.updateProjectionMatrix();else{const aspect2=width/height;this.orthographicCamera.left=-10*aspect2/2,this.orthographicCamera.right=10*aspect2/2,this.orthographicCamera.top=10/2,this.orthographicCamera.bottom=-10/2,this.orthographicCamera.updateProjectionMatrix()}this.renderer.render(this.scene,this.activeCamera);const imageData=this.renderer.domElement.toDataURL("image/png");this.renderer.setSize(originalWidth,originalHeight),this.handleResize(),resolve(imageData)}catch(error){reject(error)}})}createSTLMaterial(){return new MeshStandardMaterial({color:8421504,metalness:.1,roughness:.8,flatShading:!1,side:DoubleSide})}setViewPosition(position){const box=new Box3;let center=new Vector3,size=new Vector3;this.currentModel&&(box.setFromObject(this.currentModel),box.getCenter(center),box.getSize(size));const distance=Math.max(size.x,size.y,size.z)*2;switch(position){case"front":this.activeCamera.position.set(0,0,distance);break;case"top":this.activeCamera.position.set(0,distance,0);break;case"right":this.activeCamera.position.set(distance,0,0);break;case"isometric":this.activeCamera.position.set(distance,distance,distance);break}this.activeCamera.lookAt(center),this.controls.target.copy(center),this.controls.update()}setBackgroundColor(color){this.renderer.setClearColor(new Color(color)),this.renderer.render(this.scene,this.activeCamera)}}class Load3dAnimation extends Load3d{static{__name(this,"Load3dAnimation")}currentAnimation=null;animationActions=[];animationClips=[];selectedAnimationIndex=0;isAnimationPlaying=!1;animationSpeed=1;constructor(container){super(container)}async setupModel(model){await super.setupModel(model),this.currentAnimation&&(this.currentAnimation.stopAllAction(),this.animationActions=[]);let animations=[];model.animations?.length>0?animations=model.animations:this.originalModel&&"animations"in this.originalModel&&(animations=this.originalModel.animations),animations.length>0&&(this.animationClips=animations,model.type==="Scene"?this.currentAnimation=new AnimationMixer(model):this.currentAnimation=new AnimationMixer(this.currentModel),this.animationClips.length>0&&this.updateSelectedAnimation(0))}setAnimationSpeed(speed){this.animationSpeed=speed,this.animationActions.forEach(action=>{action.setEffectiveTimeScale(speed)})}updateSelectedAnimation(index){if(!this.currentAnimation||!this.animationClips||index>=this.animationClips.length){console.warn("Invalid animation update request");return}this.animationActions.forEach(action2=>{action2.stop()}),this.currentAnimation.stopAllAction(),this.animationActions=[],this.selectedAnimationIndex=index;const clip=this.animationClips[index],action=this.currentAnimation.clipAction(clip);action.setEffectiveTimeScale(this.animationSpeed),action.reset(),action.clampWhenFinished=!1,action.loop=LoopRepeat,this.isAnimationPlaying?action.play():(action.play(),action.paused=!0),this.animationActions=[action]}clearModel(){this.currentAnimation&&(this.animationActions.forEach(action=>{action.stop()}),this.currentAnimation=null),this.animationActions=[],this.animationClips=[],this.selectedAnimationIndex=0,this.isAnimationPlaying=!1,this.animationSpeed=1,super.clearModel()}getAnimationNames(){return this.animationClips.map((clip,index)=>clip.name||`Animation ${index+1}`)}toggleAnimation(play){if(!this.currentAnimation||this.animationActions.length===0){console.warn("No animation to toggle");return}this.isAnimationPlaying=play??!this.isAnimationPlaying,this.animationActions.forEach(action=>{this.isAnimationPlaying?(action.paused=!1,(action.time===0||action.time===action.getClip().duration)&&action.reset()):action.paused=!0})}animate=__name(()=>{if(requestAnimationFrame(this.animate),this.currentAnimation&&this.isAnimationPlaying){const delta=this.clock.getDelta();this.currentAnimation.update(delta)}this.controls.update(),this.renderer.render(this.scene,this.activeCamera)},"animate")}function splitFilePath(path){const folder_separator=path.lastIndexOf("/");return folder_separator===-1?["",path]:[path.substring(0,folder_separator),path.substring(folder_separator+1)]}__name(splitFilePath,"splitFilePath");function getResourceURL(subfolder,filename,type="input"){return`/view?${["filename="+encodeURIComponent(filename),"type="+type,"subfolder="+subfolder,app.getRandParam().substring(1)].join("&")}`}__name(getResourceURL,"getResourceURL");const load3dCSSCLASS=`display: flex;
    flex-direction: column;
    background: transparent;
    flex: 1;
    position: relative;
    overflow: hidden;`,load3dCanvasCSSCLASS=`display: flex;
    width: 100% !important;
    height: 100% !important;`,containerToLoad3D=new Map;function configureLoad3D(load3d,loadFolder,modelWidget,showGrid,cameraType,view,material,bgColor,lightIntensity,upDirection,cameraState,postModelUpdateFunc){const onModelWidgetUpdate=__name(()=>{let isFirstLoad=!0;return async value=>{if(!value)return;const filename=value,modelUrl=api.apiURL(getResourceURL(...splitFilePath(filename),loadFolder));if(await load3d.loadModel(modelUrl,filename),load3d.setMaterialMode(material.value),load3d.setUpDirection(upDirection.value),postModelUpdateFunc&&postModelUpdateFunc(load3d),isFirstLoad&&cameraState&&typeof cameraState=="object"){try{load3d.setCameraState(cameraState)}catch(error){console.warn("Failed to restore camera state:",error)}isFirstLoad=!1}}},"createModelUpdateHandler")();modelWidget.value&&onModelWidgetUpdate(modelWidget.value),modelWidget.callback=onModelWidgetUpdate,load3d.toggleGrid(showGrid.value),showGrid.callback=value=>{load3d.toggleGrid(value)},load3d.toggleCamera(cameraType.value),cameraType.callback=value=>{load3d.toggleCamera(value)},view.callback=value=>{load3d.setViewPosition(value)},material.callback=value=>{load3d.setMaterialMode(value)},load3d.setMaterialMode(material.value),load3d.setBackgroundColor(bgColor.value),bgColor.callback=value=>{load3d.setBackgroundColor(value)},load3d.setLightIntensity(lightIntensity.value),lightIntensity.callback=value=>{load3d.setLightIntensity(value)},upDirection.callback=value=>{load3d.setUpDirection(value)},load3d.setUpDirection(upDirection.value)}__name(configureLoad3D,"configureLoad3D");app.registerExtension({name:"Comfy.Load3D",getCustomWidgets(app2){return{LOAD_3D(node,inputName){let load3dNode=app2.graph._nodes.filter(wi=>wi.type=="Load3D");node.addProperty("Camera Info","");const container=document.createElement("div");container.id=`comfy-load-3d-${load3dNode.length}`,container.classList.add("comfy-load-3d");const load3d=new Load3d(container);containerToLoad3D.set(container.id,load3d),node.onResize=function(){load3d&&load3d.handleResize()};const origOnRemoved=node.onRemoved;node.onRemoved=function(){load3d&&load3d.remove(),containerToLoad3D.delete(container.id),origOnRemoved?.apply(this,[])},node.onDrawBackground=function(){load3d.renderer.domElement.hidden=this.flags.collapsed??!1};const fileInput=document.createElement("input");return fileInput.type="file",fileInput.accept=".gltf,.glb,.obj,.mtl,.fbx,.stl",fileInput.style.display="none",fileInput.onchange=async()=>{if(fileInput.files?.length){const modelWidget=node.widgets?.find(w=>w.name==="model_file"),uploadPath=await uploadFile(load3d,fileInput.files[0],fileInput).catch(error=>{console.error("File upload failed:",error),useToastStore().addAlert("File upload failed")});uploadPath&&modelWidget&&(modelWidget.options?.values?.includes(uploadPath)||modelWidget.options?.values?.push(uploadPath),modelWidget.value=uploadPath)}},node.addWidget("button","upload 3d model","upload3dmodel",()=>{fileInput.click()}),node.addWidget("button","clear","clear",()=>{load3d.clearModel();const modelWidget=node.widgets?.find(w=>w.name==="model_file");modelWidget&&(modelWidget.value="")}),{widget:node.addDOMWidget(inputName,"LOAD_3D",container)}}}},init(){const style=document.createElement("style");style.innerText=`
        .comfy-load-3d {
          ${load3dCSSCLASS}
        }
        
        .comfy-load-3d canvas {
          ${load3dCanvasCSSCLASS}
        }
      `,document.head.appendChild(style)},async nodeCreated(node){if(node.constructor.comfyClass!=="Load3D")return;const[oldWidth,oldHeight]=node.size;node.setSize([Math.max(oldWidth,300),Math.max(oldHeight,600)]),await nextTick();const sceneWidget=node.widgets.find(w2=>w2.name==="image"),container=sceneWidget.element,load3d=containerToLoad3D.get(container.id),modelWidget=node.widgets.find(w2=>w2.name==="model_file"),showGrid=node.widgets.find(w2=>w2.name==="show_grid"),cameraType=node.widgets.find(w2=>w2.name==="camera_type"),view=node.widgets.find(w2=>w2.name==="view"),material=node.widgets.find(w2=>w2.name==="material"),bgColor=node.widgets.find(w2=>w2.name==="bg_color"),lightIntensity=node.widgets.find(w2=>w2.name==="light_intensity"),upDirection=node.widgets.find(w2=>w2.name==="up_direction");let cameraState;try{const cameraInfo=node.properties["Camera Info"];cameraInfo&&typeof cameraInfo=="string"&&cameraInfo.trim()!==""&&(cameraState=JSON.parse(cameraInfo))}catch(error){console.warn("Failed to parse camera state:",error),cameraState=void 0}configureLoad3D(load3d,"input",modelWidget,showGrid,cameraType,view,material,bgColor,lightIntensity,upDirection,cameraState);const w=node.widgets.find(w2=>w2.name==="width"),h=node.widgets.find(w2=>w2.name==="height");sceneWidget.serializeValue=async()=>{node.properties["Camera Info"]=JSON.stringify(load3d.getCameraState());const imageData=await load3d.captureScene(w.value,h.value),blob=await fetch(imageData).then(r=>r.blob()),name=`scene_${Date.now()}.png`,file2=new File([blob],name),body=new FormData;body.append("image",file2),body.append("subfolder","threed"),body.append("type","temp");const resp=await api.fetchApi("/upload/image",{method:"POST",body});if(resp.status!==200){const err2=`Error uploading scene capture: ${resp.status} - ${resp.statusText}`;throw useToastStore().addAlert(err2),new Error(err2)}return`threed/${(await resp.json()).name} [temp]`}}});app.registerExtension({name:"Comfy.Load3DAnimation",getCustomWidgets(app2){return{LOAD_3D_ANIMATION(node,inputName){let load3dNode=app2.graph._nodes.filter(wi=>wi.type=="Load3DAnimation");node.addProperty("Camera Info","");const container=document.createElement("div");container.id=`comfy-load-3d-animation-${load3dNode.length}`,container.classList.add("comfy-load-3d-animation");const load3d=new Load3dAnimation(container);containerToLoad3D.set(container.id,load3d),node.onResize=function(){load3d&&load3d.handleResize()};const origOnRemoved=node.onRemoved;node.onRemoved=function(){load3d&&load3d.remove(),containerToLoad3D.delete(container.id),origOnRemoved?.apply(this,[])},node.onDrawBackground=function(){load3d.renderer.domElement.hidden=this.flags.collapsed??!1};const fileInput=document.createElement("input");fileInput.type="file",fileInput.accept=".fbx,glb,gltf",fileInput.style.display="none",fileInput.onchange=async()=>{if(fileInput.files?.length){const modelWidget=node.widgets?.find(w=>w.name==="model_file"),uploadPath=await uploadFile(load3d,fileInput.files[0],fileInput).catch(error=>{console.error("File upload failed:",error),useToastStore().addAlert("File upload failed")});uploadPath&&modelWidget&&(modelWidget.options?.values?.includes(uploadPath)||modelWidget.options?.values?.push(uploadPath),modelWidget.value=uploadPath)}},node.addWidget("button","upload 3d model","upload3dmodel",()=>{fileInput.click()}),node.addWidget("button","clear","clear",()=>{load3d.clearModel();const modelWidget=node.widgets?.find(w=>w.name==="model_file");modelWidget&&(modelWidget.value="");const animationSelect2=node.widgets?.find(w=>w.name==="animation");animationSelect2&&(animationSelect2.options.values=[],animationSelect2.value="");const speedSelect=node.widgets?.find(w=>w.name==="animation_speed");speedSelect&&(speedSelect.value="1")}),node.addWidget("button","Play/Pause Animation","toggle_animation",()=>{load3d.toggleAnimation()});const animationSelect=node.addWidget("combo","animation","",()=>"",{values:[]});return animationSelect.callback=value=>{const index=load3d.getAnimationNames().indexOf(value);if(index!==-1){const wasPlaying=load3d.isAnimationPlaying;wasPlaying&&load3d.toggleAnimation(!1),load3d.updateSelectedAnimation(index),wasPlaying&&load3d.toggleAnimation(!0)}},{widget:node.addDOMWidget(inputName,"LOAD_3D_ANIMATION",container)}}}},init(){const style=document.createElement("style");style.innerText=`
        .comfy-load-3d-animation {
          ${load3dCSSCLASS}
        }
        
        .comfy-load-3d-animation canvas {
          ${load3dCanvasCSSCLASS}
        }
      `,document.head.appendChild(style)},async nodeCreated(node){if(node.constructor.comfyClass!=="Load3DAnimation")return;const[oldWidth,oldHeight]=node.size;node.setSize([Math.max(oldWidth,300),Math.max(oldHeight,700)]),await nextTick();const sceneWidget=node.widgets.find(w2=>w2.name==="image"),container=sceneWidget.element,load3d=containerToLoad3D.get(container.id),modelWidget=node.widgets.find(w2=>w2.name==="model_file"),showGrid=node.widgets.find(w2=>w2.name==="show_grid"),cameraType=node.widgets.find(w2=>w2.name==="camera_type"),view=node.widgets.find(w2=>w2.name==="view"),material=node.widgets.find(w2=>w2.name==="material"),bgColor=node.widgets.find(w2=>w2.name==="bg_color"),lightIntensity=node.widgets.find(w2=>w2.name==="light_intensity"),upDirection=node.widgets.find(w2=>w2.name==="up_direction"),speedSelect=node.widgets.find(w2=>w2.name==="animation_speed");speedSelect.callback=value=>{const load3d2=containerToLoad3D.get(container.id);load3d2&&load3d2.setAnimationSpeed(parseFloat(value))};let cameraState;try{const cameraInfo=node.properties["Camera Info"];cameraInfo&&typeof cameraInfo=="string"&&cameraInfo.trim()!==""&&(cameraState=JSON.parse(cameraInfo))}catch(error){console.warn("Failed to parse camera state:",error),cameraState=void 0}configureLoad3D(load3d,"input",modelWidget,showGrid,cameraType,view,material,bgColor,lightIntensity,upDirection,cameraState,load3d2=>{const names=load3d2.getAnimationNames(),animationSelect=node.widgets.find(w2=>w2.name==="animation");animationSelect.options.values=names,names.length&&(animationSelect.value=names[0])});const w=node.widgets.find(w2=>w2.name==="width"),h=node.widgets.find(w2=>w2.name==="height");sceneWidget.serializeValue=async()=>{node.properties["Camera Info"]=JSON.stringify(load3d.getCameraState()),load3d.toggleAnimation(!1);const imageData=await load3d.captureScene(w.value,h.value),blob=await fetch(imageData).then(r=>r.blob()),name=`scene_${Date.now()}.png`,file2=new File([blob],name),body=new FormData;body.append("image",file2),body.append("subfolder","threed"),body.append("type","temp");const resp=await api.fetchApi("/upload/image",{method:"POST",body});if(resp.status!==200){const err2=`Error uploading scene capture: ${resp.status} - ${resp.statusText}`;throw useToastStore().addAlert(err2),new Error(err2)}return`threed/${(await resp.json()).name} [temp]`}}});app.registerExtension({name:"Comfy.Preview3D",getCustomWidgets(app2){return{PREVIEW_3D(node,inputName){let load3dNode=app2.graph._nodes.filter(wi=>wi.type=="Preview3D");const container=document.createElement("div");container.id=`comfy-preview-3d-${load3dNode.length}`,container.classList.add("comfy-preview-3d");const load3d=new Load3d(container);containerToLoad3D.set(container.id,load3d),node.onResize=function(){load3d&&load3d.handleResize()};const origOnRemoved=node.onRemoved;return node.onRemoved=function(){load3d&&load3d.remove(),containerToLoad3D.delete(container.id),origOnRemoved?.apply(this,[])},node.onDrawBackground=function(){load3d.renderer.domElement.hidden=this.flags.collapsed??!1},{widget:node.addDOMWidget(inputName,"PREVIEW_3D",container)}}}},init(){const style=document.createElement("style");style.innerText=`
        .comfy-preview-3d {
          ${load3dCSSCLASS}
        }
        
        .comfy-preview-3d canvas {
          ${load3dCanvasCSSCLASS}
        }
      `,document.head.appendChild(style)},async nodeCreated(node){if(node.constructor.comfyClass!=="Preview3D")return;const[oldWidth,oldHeight]=node.size;node.setSize([Math.max(oldWidth,300),Math.max(oldHeight,550)]),await nextTick();const container=node.widgets.find(w=>w.name==="image").element,load3d=containerToLoad3D.get(container.id),modelWidget=node.widgets.find(w=>w.name==="model_file"),showGrid=node.widgets.find(w=>w.name==="show_grid"),cameraType=node.widgets.find(w=>w.name==="camera_type"),view=node.widgets.find(w=>w.name==="view"),material=node.widgets.find(w=>w.name==="material"),bgColor=node.widgets.find(w=>w.name==="bg_color"),lightIntensity=node.widgets.find(w=>w.name==="light_intensity"),upDirection=node.widgets.find(w=>w.name==="up_direction");configureLoad3D(load3d,"output",modelWidget,showGrid,cameraType,view,material,bgColor,lightIntensity,upDirection)}});
//# sourceMappingURL=index-BQTuXuWE.js.map