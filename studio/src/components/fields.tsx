import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import { ChevronIcon } from "./icons";

interface NumberFieldProps {
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  onCommit(value: number): void;
}

/**
 * Numeric input that commits on blur or Enter and reverts on Escape.
 *
 * Editing must not fire a command per keystroke: that would flood the undo
 * stack and autosave with half-typed values.
 */
export function NumberField({
  label,
  value,
  step = 0.1,
  min,
  max,
  disabled,
  onCommit,
}: NumberFieldProps) {
  const id = useId();
  const [draft, setDraft] = useState(() => format(value));
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setDraft(format(value));
  }, [value]);

  const commit = () => {
    focused.current = false;
    const parsed = Number.parseFloat(draft);
    if (!Number.isFinite(parsed)) {
      setDraft(format(value));
      return;
    }
    const clamped = Math.min(max ?? Number.POSITIVE_INFINITY, Math.max(min ?? Number.NEGATIVE_INFINITY, parsed));
    if (Math.abs(clamped - value) > 1e-9) onCommit(clamped);
    setDraft(format(clamped));
    if (clamped !== parsed) {
      // Show the value the model actually accepted.
      setDraft(format(clamped));
    }
  };

  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{label}</span>
      <input
        id={id}
        className="input input--number"
        type="number"
        inputMode="decimal"
        step={step}
        min={min}
        max={max}
        disabled={disabled}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onFocus={() => {
          focused.current = true;
        }}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          } else if (event.key === "Escape") {
            focused.current = false;
            setDraft(format(value));
            event.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}

interface RangeFieldProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  onCommit(value: number): void;
}

/**
 * Slider that previews while dragging and commits once on release.
 *
 * Committing per input event would push one undo entry per pixel of the drag;
 * the published value is what the model accepts, and the drag still reads
 * live in the label. Keyboard use commits on key-up.
 */
export function RangeField({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  disabled,
  onCommit,
}: RangeFieldProps) {
  const id = useId();
  const [draft, setDraft] = useState(value);
  const dragging = useRef(false);

  useEffect(() => {
    // A model update (undo, project open) wins once the pointer is released.
    if (!dragging.current) setDraft(value);
  }, [value]);

  const publish = (next: number) => {
    const clamped = Math.min(max, Math.max(min, Number.isFinite(next) ? next : value));
    setDraft(clamped);
    if (Math.abs(clamped - value) > 1e-9) onCommit(clamped);
  };

  return (
    <label className="field field--range" htmlFor={id}>
      <span className="field-label">{label}</span>
      <div className="field-range">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={draft}
          disabled={disabled}
          onPointerDown={() => {
            dragging.current = true;
          }}
          onChange={(event) => setDraft(Number(event.target.value))}
          onPointerUp={(event) => {
            dragging.current = false;
            publish(Number(event.currentTarget.value));
          }}
          onKeyUp={(event) => publish(Number(event.currentTarget.value))}
          onBlur={(event) => {
            dragging.current = false;
            publish(Number(event.currentTarget.value));
          }}
        />
        <span className="field-range__value mono">{draft.toFixed(2)}</span>
      </div>
    </label>
  );
}

interface TextFieldProps {
  label: string;
  value: string;
  disabled?: boolean;
  onCommit(value: string): void;
}
export function TextField({ label, value, disabled, onCommit }: TextFieldProps) {
  const id = useId();
  const [draft, setDraft] = useState(value);
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setDraft(value);
  }, [value]);

  const commit = () => {
    focused.current = false;
    const trimmed = draft.trim();
    if (!trimmed || trimmed === value) {
      setDraft(value);
      return;
    }
    onCommit(trimmed);
  };

  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{label}</span>
      <input
        id={id}
        className="input"
        type="text"
        value={draft}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        onFocus={() => {
          focused.current = true;
        }}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
          if (event.key === "Escape") {
            focused.current = false;
            setDraft(value);
            event.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}

/**
 * A properties panel. Collapsible, like every Blender panel: the header is the
 * toggle, and an `id` remembers the state between runs (a settings sidebar whose
 * sections forget their state is a settings sidebar you re-open by hand).
 */
export function Section({
  title,
  children,
  actions,
  id,
  defaultOpen = true,
}: {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
  /** Stable key for the remembered open/closed state. */
  id?: string;
  defaultOpen?: boolean;
}) {
  const storageKey = id ? `studio.section.${id}` : null;
  const [open, setOpen] = useState(() => {
    if (!storageKey) return defaultOpen;
    try {
      const stored = window.localStorage.getItem(storageKey);
      return stored === null ? defaultOpen : stored === "1";
    } catch {
      return defaultOpen;
    }
  });

  const toggle = () => {
    setOpen((value) => {
      const next = !value;
      if (storageKey) {
        try {
          window.localStorage.setItem(storageKey, next ? "1" : "0");
        } catch {
          // Storage disabled: the section still toggles, it just forgets.
        }
      }
      return next;
    });
  };

  return (
    <section className={`inspector-section${open ? "" : " is-collapsed"}`}>
      <header className="inspector-section__header">
        <button
          type="button"
          className="inspector-section__toggle"
          aria-expanded={open}
          onClick={toggle}
        >
          <ChevronIcon open={open} size={12} />
          <h3>{title}</h3>
        </button>
        {actions}
      </header>
      {open && <div className="inspector-section__body">{children}</div>}
    </section>
  );
}

/** A small segmented control — the two-or-three option cases (a select is
 *  heavier and hides the alternatives). */
export function ChoiceField<T extends string>({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: T;
  options: { value: T; label: string; title?: string; disabled?: boolean }[];
  onChange(value: T): void;
  hint?: string;
}) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <div className="segmented" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`segmented__option${option.value === value ? " is-active" : ""}`}
            aria-pressed={option.value === value}
            title={option.title ?? option.label}
            disabled={option.disabled}
            onClick={() => {
              if (option.value !== value) onChange(option.value);
            }}
          >
            {option.label}
          </button>
        ))}
      </div>
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange(checked: boolean): void;
}) {
  const id = useId();
  return (
    <label className="toggle" htmlFor={id}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="toggle__track" aria-hidden="true">
        <span className="toggle__thumb" />
      </span>
      <span className="toggle__label">{label}</span>
    </label>
  );
}

function format(value: number): string {
  if (!Number.isFinite(value)) return "0";
  const rounded = Math.round(value * 1000) / 1000;
  return Object.is(rounded, -0) ? "0" : String(rounded);
}
