"use client";

import { Button } from "./ui/button";
import { GitIcon, VercelIcon } from "./icons";
import Link from "next/link";

export const Navbar = () => {
  return (
    <div className="px-3 py-2 flex flex-row gap-2 justify-between border-b">
      <Link href="https://cogniformai.com" className="text-xl font-medium">
        CogniformAI
      </Link>
      <Button size={"sm"} variant="ghost">
        New Chat
      </Button>
    </div>
  );
};
