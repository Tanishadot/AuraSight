import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useId } from "react";

type Variant = "primary" | "secondary" | "quiet" | "danger";

const variants: Record<Variant, string> = {
  primary:
    "bg-sage text-sage-contrast hover:brightness-110 active:brightness-95 border border-transparent shadow-sm",
  secondary: "bg-surface text-ink border border-line-strong hover:bg-surface-2",
  quiet: "bg-transparent text-ink border border-transparent hover:bg-surface-2",
  danger: "bg-danger-soft text-danger border border-danger/40 hover:brightness-95",
};

export function Button({
  variant = "secondary",
  size = "md",
  icon,
  shortcut,
  children,
  className = "",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: "md" | "lg" | "icon";
  icon?: ReactNode;
  shortcut?: string;
}) {
  const sizes = {
    md: "min-h-11 px-4 text-[0.95rem] gap-2",
    lg: "min-h-14 px-5 text-lg gap-3",
    icon: "h-11 w-11 justify-center",
  };
  return (
    <button
      type="button"
      aria-keyshortcuts={shortcut}
      className={`inline-flex items-center whitespace-nowrap rounded-xl font-bold transition-[filter,background-color] duration-150 disabled:opacity-45 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`}
      {...rest}
    >
      {icon}
      {children}
      {shortcut && size !== "icon" && (
        <kbd className="ml-auto pl-2 font-mono text-xs font-normal opacity-70" aria-hidden="true">
          {shortcut}
        </kbd>
      )}
    </button>
  );
}

export function Card({
  title,
  icon,
  actions,
  children,
  className = "",
  labelledBy,
}: {
  title?: string;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  labelledBy?: string;
}) {
  const id = useId();
  const headingId = labelledBy ?? `${id}-h`;
  return (
    <section
      aria-labelledby={title ? headingId : undefined}
      className={`rounded-2xl border border-line bg-surface p-5 shadow-[0_1px_0_rgba(0,0,0,0.03)] ${className}`}
    >
      {title && (
        <div className="mb-4 flex items-center gap-3">
          {icon && <span className="text-sage-ink" aria-hidden="true">{icon}</span>}
          <h2 id={headingId} className="text-[1.05rem] font-bold tracking-tight">
            {title}
          </h2>
          {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Toggle({
  label,
  description,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <div>
        <label htmlFor={id} className="font-bold">
          {label}
        </label>
        {description && (
          <p id={`${id}-d`} className="text-sm text-muted">
            {description}
          </p>
        )}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-describedby={description ? `${id}-d` : undefined}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 inline-flex h-8 w-14 shrink-0 items-center rounded-full border transition-colors disabled:opacity-45 ${
          checked ? "border-sage bg-sage" : "border-line-strong bg-surface-2"
        }`}
      >
        <span
          className={`inline-block h-6 w-6 rounded-full shadow transition-transform ${
            checked ? "translate-x-7 bg-sage-contrast" : "translate-x-1 bg-muted"
          }`}
        />
        <span className="sr-only">{checked ? "On" : "Off"}</span>
      </button>
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  const id = useId();
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="font-bold">
          {label}
        </label>
        <span className="tabular text-sm text-muted" aria-hidden="true">
          {format(value)}
        </span>
      </div>
      <input
        id={id}
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-valuetext={format(value)}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function Select<T extends string | number>({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  hint?: string;
}) {
  const id = useId();
  return (
    <div className="py-1.5">
      <label htmlFor={id} className="block font-bold">
        {label}
      </label>
      {hint && <p className="text-sm text-muted">{hint}</p>}
      <select
        id={id}
        value={String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          const opt = options.find((o) => String(o.value) === raw);
          if (opt) onChange(opt.value);
        }}
        className="mt-1 min-h-11 w-full rounded-xl border border-line-strong bg-surface px-3 text-ink"
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function Badge({ tone = "neutral", children }: { tone?: "neutral" | "sage" | "warn" | "danger"; children: ReactNode }) {
  const tones = {
    neutral: "bg-surface-2 text-muted border-line",
    sage: "bg-sage-soft text-sage-ink border-sage/30",
    warn: "bg-warn-soft text-warn border-warn/30",
    danger: "bg-danger-soft text-danger border-danger/30",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-bold ${tones[tone]}`}>
      {children}
    </span>
  );
}
