var __defProp=Object.defineProperty;var __name=(target,value)=>__defProp(target,"name",{value,configurable:!0});import{d as defineComponent,bS as useExtensionStore,u as useSettingStore,r as ref,o as onMounted,q as computed,g as openBlock,h as createElementBlock,i as createVNode,y as withCtx,z as unref,bT as script$1,A as createBaseVNode,x as createBlock,N as Fragment,O as renderList,a6 as toDisplayString,aw as createTextVNode,j as createCommentVNode,D as script$4}from"./index-7Cm1luhB.js";import{s as script,a as script$2,b as script$3}from"./index-2dY2q85P.js";import"./index-Bs5CNKKh.js";const _hoisted_1={class:"extension-panel"},_hoisted_2={class:"mt-4"},_sfc_main=defineComponent({__name:"ExtensionPanel",setup(__props){const extensionStore=useExtensionStore(),settingStore=useSettingStore(),editingEnabledExtensions=ref({});onMounted(()=>{extensionStore.extensions.forEach(ext=>{editingEnabledExtensions.value[ext.name]=extensionStore.isExtensionEnabled(ext.name)})});const changedExtensions=computed(()=>extensionStore.extensions.filter(ext=>editingEnabledExtensions.value[ext.name]!==extensionStore.isExtensionEnabled(ext.name))),hasChanges=computed(()=>changedExtensions.value.length>0),updateExtensionStatus=__name(()=>{const editingDisabledExtensionNames=Object.entries(editingEnabledExtensions.value).filter(([_,enabled])=>!enabled).map(([name])=>name);settingStore.set("Comfy.Extension.Disabled",[...extensionStore.inactiveDisabledExtensionNames,...editingDisabledExtensionNames])},"updateExtensionStatus"),applyChanges=__name(()=>{window.location.reload()},"applyChanges");return(_ctx,_cache)=>(openBlock(),createElementBlock("div",_hoisted_1,[createVNode(unref(script$2),{value:unref(extensionStore).extensions,stripedRows:"",size:"small"},{default:withCtx(()=>[createVNode(unref(script),{field:"name",header:_ctx.$t("extensionName"),sortable:""},null,8,["header"]),createVNode(unref(script),{pt:{bodyCell:"flex items-center justify-end"}},{body:withCtx(slotProps=>[createVNode(unref(script$1),{modelValue:editingEnabledExtensions.value[slotProps.data.name],"onUpdate:modelValue":__name($event=>editingEnabledExtensions.value[slotProps.data.name]=$event,"onUpdate:modelValue"),onChange:updateExtensionStatus},null,8,["modelValue","onUpdate:modelValue"])]),_:1})]),_:1},8,["value"]),createBaseVNode("div",_hoisted_2,[hasChanges.value?(openBlock(),createBlock(unref(script$3),{key:0,severity:"info"},{default:withCtx(()=>[createBaseVNode("ul",null,[(openBlock(!0),createElementBlock(Fragment,null,renderList(changedExtensions.value,ext=>(openBlock(),createElementBlock("li",{key:ext.name},[createBaseVNode("span",null,toDisplayString(unref(extensionStore).isExtensionEnabled(ext.name)?"[-]":"[+]"),1),createTextVNode(" "+toDisplayString(ext.name),1)]))),128))])]),_:1})):createCommentVNode("",!0),createVNode(unref(script$4),{label:_ctx.$t("reloadToApplyChanges"),icon:"pi pi-refresh",onClick:applyChanges,disabled:!hasChanges.value,text:"",fluid:"",severity:"danger"},null,8,["label","disabled"])])]))}});export{_sfc_main as default};
//# sourceMappingURL=ExtensionPanel-C0fZTn1F.js.map