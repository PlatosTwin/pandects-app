import logo128Webp from "../../assets/logo-128.webp";
import logo256Webp from "../../assets/logo-256.webp";
import logo128Png from "../../assets/logo-128.png";
import logo256Png from "../../assets/logo-256.png";

export function NavLogo({ pandaTarget }: { pandaTarget?: string }) {
  return (
    <picture>
      <source
        srcSet={`${logo128Webp} 128w, ${logo256Webp} 256w`}
        sizes="36px"
        type="image/webp"
      />
      <img
        src={logo128Png}
        alt="Pandects Logo"
        data-panda-target={pandaTarget}
        width={36}
        height={36}
        srcSet={`${logo128Png} 128w, ${logo256Png} 256w`}
        sizes="36px"
        decoding="async"
        className="relative z-10 h-9 w-9 rounded-md object-cover ring-1 ring-border"
      />
    </picture>
  );
}
