// Значки — тонкие линии 24×24, рисованные для приложения.
const s = (body) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;

export const icons = {
  data: s('<path d="M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Z"/><path d="M4 7v5c0 1.7 3.6 3 8 3s8-1.3 8-3V7"/><path d="M4 12v5c0 1.7 3.6 3 8 3s8-1.3 8-3v-5"/>'),
  align: s('<path d="M7 4h4v4H7zM13 16h4v4h-4z"/><path d="M9 8v3a2 2 0 0 0 2 2h4a2 2 0 0 1 2 2v1"/><path d="m4 20 4-4M20 4l-4 4"/>'),
  layers: s('<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/>'),
  export: s('<path d="M12 3v12"/><path d="m7 8 5-5 5 5"/><path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/>'),
  ct: s('<rect x="3.5" y="3.5" width="17" height="17" rx="4"/><circle cx="12" cy="12" r="4.5"/><path d="M12 3.5v4M12 16.5v4"/>'),
  tooth: s('<path d="M7.5 3.5c-2.5 0-4 2-4 4.5 0 3 1.6 4.2 2.3 7.1.6 2.6 1 5.4 2.6 5.4 1.8 0 1.6-4.5 3.6-4.5s1.8 4.5 3.6 4.5c1.6 0 2-2.8 2.6-5.4.7-2.9 2.3-4.1 2.3-7.1 0-2.5-1.5-4.5-4-4.5-2 0-2.8 1.2-4.5 1.2S9.5 3.5 7.5 3.5Z"/>'),
  plus: s('<path d="M12 5v14M5 12h14"/>'),
  folder: s('<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2Z"/>'),
  file: s('<path d="M14 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5Z"/><path d="M14 3.5v5h5"/>'),
  trash: s('<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 12.5h9l1-12.5"/>'),
  play: s('<path d="M7 4.5v15l12-7.5-12-7.5Z"/>'),
  check: s('<path d="m5 12.5 4.5 4.5L19 7.5"/>'),
  refine: s('<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.5"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'),
  undo: s('<path d="M9 7 4.5 11.5 9 16"/><path d="M5 11.5h9a5 5 0 0 1 0 10h-2"/>'),
  move: s('<path d="M12 3v18M3 12h18"/><path d="m9 6 3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>'),
  rotate: s('<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4.5v4h-4"/>'),
  eye: s('<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="3"/>'),
  heat: s('<path d="M12 3c3 4 5 6.2 5 9.5a5 5 0 0 1-10 0C7 9.2 9 7 12 3Z"/><path d="M12 13.5a1.8 1.8 0 0 0 1.8-1.8"/>'),
  front: s('<rect x="5" y="5" width="14" height="14" rx="3"/><circle cx="12" cy="12" r="1.5"/>'),
  side: s('<path d="M5 5h14v14H5z"/><path d="M5 12h14"/>'),
  top: s('<path d="M5 5h14v14H5z"/><path d="M12 5v14"/>'),
  warn: s('<path d="M12 4 2.5 20h19L12 4Z"/><path d="M12 10v4.5M12 17.5v.01"/>'),
  chevron: s('<path d="m9 6 6 6-6 6"/>'),
};
