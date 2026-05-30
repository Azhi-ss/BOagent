import { AcqSelect, Field, NumberField } from "./Field";
import type { AcquisitionType } from "../types";

interface AcquisitionConfigProps {
  config: {
    acquisition: AcquisitionType;
    xi: number;
    kappa: number;
  };
  onChange: (updates: Partial<AcquisitionConfigProps["config"]>) => void;
  accent: string;
}

export function AcquisitionConfig({ config, onChange, accent }: AcquisitionConfigProps) {
  return (
    <>
      <Field label="采集函数" subLabel="Acquisition Function">
        <AcqSelect
          value={config.acquisition}
          onChange={(v) => onChange({ acquisition: v })}
          accent={accent}
        />
      </Field>
      
      <div style={{ display: "flex", gap: 16 }}>
        {(config.acquisition === "ei" || config.acquisition === "pi") && (
          <Field label="探索余长" subLabel="Exploration margin ξ (Xi)" hint="调节寻找“惊喜”的阈值">
            <NumberField 
              value={config.xi} 
              onChange={(v) => onChange({ xi: v })} 
              step={0.01} 
              min={0} 
              width={100} 
            />
          </Field>
        )}
        {config.acquisition === "ucb" && (
          <Field label="置信系数" subLabel="Confidence width κ (Kappa)" hint="越大越冒险，越小越稳健">
            <NumberField 
              value={config.kappa} 
              onChange={(v) => onChange({ kappa: v })} 
              step={0.1} 
              min={0} 
              width={100} 
            />
          </Field>
        )}
      </div>
    </>
  );
}
