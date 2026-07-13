"use client";

import { useId, useState, type ReactNode } from "react";

/**
 * A smoothly-animating disclosure. Unlike native <details> (which drops its
 * content when closed and so can't transition open OR closed), this keeps the
 * content in the DOM and animates it with the grid-template-rows 0fr↔1fr
 * technique — no height/layout animation, just a compositor-friendly reveal.
 * Under prefers-reduced-motion the transition is neutralized globally, so it
 * toggles instantly. Semantics match a disclosure: a real <button> with
 * aria-expanded controlling a region.
 */
export default function Collapsible({
  label,
  children,
  defaultOpen = false,
  rootClassName = "",
  summaryClassName = "",
  contentClassName = "",
}: {
  label: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  rootClassName?: string;
  summaryClassName?: string;
  contentClassName?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();

  return (
    <div className={rootClassName}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className={`${summaryClassName} focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/60`}
      >
        <span
          aria-hidden
          className="text-faint transition-transform duration-200 motion-reduce:transition-none"
          style={{ transform: open ? "rotate(90deg)" : "none" }}
        >
          ›
        </span>
        {label}
      </button>
      <div
        id={id}
        // grid-template-rows 0fr → 1fr: the accepted, non-layout-thrashing way to
        // animate a disclosure open AND closed while the content stays mounted.
        className="grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className={`min-h-0 overflow-hidden ${contentClassName}`}>{children}</div>
      </div>
    </div>
  );
}
