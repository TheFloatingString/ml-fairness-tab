import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tabular hardware-fairness results",
  description:
    "Accuracy spread across datasets, models, GPUs and float types for the tabular hardware-fairness experiment.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
