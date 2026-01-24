import type { Metadata } from "next";
import { DM_Sans, Fraunces } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Navigation } from "@/components/Navigation";

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "PolicyEngine UK Targets Explorer",
  description: "Explore calibration targets and official statistics for UK tax-benefit modelling",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${dmSans.variable} ${fraunces.variable} antialiased min-h-screen`}>
        <Providers>
          <div className="flex min-h-screen">
            <Navigation />
            <main className="flex-1 ml-64">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
