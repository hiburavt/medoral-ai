import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { ClipboardList } from 'lucide-react';

export interface BioData {
  age: string;
  gender: string;
  tobacco: string;
  alcohol: string;
  symptoms_duration: string;
  pain_level: string;
}

interface BioFormProps {
  onComplete: (data: BioData) => void;
}

export default function BioForm({ onComplete }: BioFormProps) {
  const [formData, setFormData] = useState<BioData>({
    age: '',
    gender: 'Male',
    tobacco: 'No',
    alcohol: 'No',
    symptoms_duration: '',
    pain_level: '0',
  });

  const handleValueChange = (name: string, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onComplete(formData);
  };

  return (
    <Card className="max-w-md mx-auto shadow-lg border-border/50">
      <CardHeader>
        <CardTitle className="text-2xl font-bold flex items-center gap-2">
          <div className="bg-primary/10 text-primary p-2 rounded-lg">
            <ClipboardList className="w-6 h-6" />
          </div>
          Patient Assessment
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="age">Age</Label>
            <Input
              id="age"
              type="number"
              name="age"
              required
              placeholder="e.g. 45"
              value={formData.age}
              onChange={handleChange}
            />
          </div>

          <div className="space-y-2">
            <Label>Gender</Label>
            <Select 
                value={formData.gender} 
                onValueChange={(val) => handleValueChange('gender', val)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Male">Male</SelectItem>
                <SelectItem value="Female">Female</SelectItem>
                <SelectItem value="Other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Tobacco Use?</Label>
              <Select 
                value={formData.tobacco} 
                onValueChange={(val) => handleValueChange('tobacco', val)}
              >
                <SelectTrigger>
                    <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="No">No</SelectItem>
                  <SelectItem value="Yes">Yes</SelectItem>
                  <SelectItem value="Past User">Past User</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Alcohol?</Label>
              <Select 
                value={formData.alcohol} 
                onValueChange={(val) => handleValueChange('alcohol', val)}
              >
                <SelectTrigger>
                    <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="No">No</SelectItem>
                  <SelectItem value="Yes">Yes</SelectItem>
                  <SelectItem value="Occasional">Occasional</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="symptoms_duration">Duration of Symptoms</Label>
            <Input
              id="symptoms_duration"
              type="text"
              name="symptoms_duration"
              required
              placeholder="e.g. 2 weeks"
              value={formData.symptoms_duration}
              onChange={handleChange}
            />
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
                <Label>Pain Level</Label>
                <span className="text-sm font-medium bg-secondary px-2 py-1 rounded-md">{formData.pain_level}/10</span>
            </div>
            <Slider 
                min={0} 
                max={10} 
                step={1} 
                value={[parseInt(formData.pain_level)]} 
                onValueChange={(val) => handleValueChange('pain_level', val[0].toString())}
                className="py-4"
            />
          </div>

          <Button type="submit" className="w-full text-lg py-6 mt-4 shadow-md hover:shadow-lg transition-all">
            Next Step: Upload Image →
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
