// Every piece of wording the station shows that is not the operator's own data lives here, so the plain-language
// version is decided in one place. The station's internal notes are mapped to words an operator uses.
(function () {
  const REASONS = [
    [/^default for [a-z_ ]+ \(no golden-board measurement\)/i, "size from the detector box"],
    [/no mm measurement: no neighbouring conductor inside the measured window/i, "no reference measurement here"],
    [/^[a-z_ ]+ is a functional defect: rejected at every IPC class/i, "rejected at every IPC class"],
    [/design rules are generic defaults, set them from the Gerber/i, "limits are the generic defaults for this design"],
  ];

  function plainReason(text) {
    if (!text) return "";
    return String(text).split("; ").map((part) => {
      for (const [re, plain] of REASONS) if (re.test(part)) return plain;
      return part;
    }).join("; ");
  }

  function modelName(file) {
    const m = /^llama3_2_(\d)b/i.exec(String(file || ""));
    return m ? `Llama 3.2 ${m[1]}B` : (file || "");
  }

  // short names for one or several findings of a type, for grouped advice ("Shorts 1, 3")
  const TYPES = {
    open: ["Open", "Opens"], short: ["Short", "Shorts"], mouse_bite: ["Mouse bite", "Mouse bites"],
    spur: ["Spur", "Spurs"], spurious_copper: ["Copper island", "Copper islands"], pin_hole: ["Pinhole", "Pinholes"],
    missing_hole: ["Missing hole", "Missing holes"], missing_pad: ["Missing pad", "Missing pads"],
    hole_breakout: ["Hole breakout", "Hole breakouts"], scratch: ["Scratch", "Scratches"],
    foreign_object: ["Foreign object", "Foreign objects"], unknown_anomaly: ["Unusual area", "Unusual areas"],
    missing_part: ["Missing part", "Missing parts"], extra_part: ["Extra part", "Extra parts"],
    changed_part: ["Changed part", "Changed parts"],
  };
  function typeName(cls, n = 1) {
    const t = TYPES[cls];
    if (t) return n > 1 ? t[1] : t[0];
    const w = String(cls || "").replaceAll("_", " ");
    return w.charAt(0).toUpperCase() + w.slice(1);
  }

  const VERDICT = {
    PASS: { word: "PASS", line: "Board passed" },
    REVIEW: { word: "CHECK", line: "look at this board" },
    FAIL: { word: "FAIL", line: "board failed" },
  };

  const PLACE = { "top-left": "top left", "top-right": "top right", "bottom-left": "bottom left",
    "bottom-right": "bottom right", "top-centre": "top", "bottom-centre": "bottom", "middle-left": "left",
    "middle-right": "right", centre: "centre" };
  const place = (loc) => PLACE[loc] || String(loc || "").replaceAll("-", " ");

  window.Labels = { plainReason, modelName, typeName, VERDICT, place };
})();
